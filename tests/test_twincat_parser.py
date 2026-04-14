"""Tests for TwinCAT Structured Text parser.

Tests the TwinCAT parser (`chunkhound/parsers/twincat/`) which handles
TcPOU XML files containing IEC 61131-3 Structured Text code.

These tests use the UniversalChunk API (extract_universal_chunks) which
produces language-agnostic chunks suitable for the universal parser pipeline.
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.twincat.twincat_parser import TwinCATParser
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept

# =============================================================================
# Test Helpers
# =============================================================================

# Mapping from ChunkType to (UniversalConcept, metadata["kind"]) for filtering
_CHUNK_TYPE_MAPPING = {
    ChunkType.PROGRAM: (UniversalConcept.DEFINITION, "program"),
    ChunkType.FUNCTION_BLOCK: (UniversalConcept.DEFINITION, "function_block"),
    ChunkType.FUNCTION: (UniversalConcept.DEFINITION, "function"),
    ChunkType.METHOD: (UniversalConcept.DEFINITION, "method"),
    ChunkType.ACTION: (UniversalConcept.DEFINITION, "action"),
    ChunkType.PROPERTY: (UniversalConcept.DEFINITION, "property"),
    ChunkType.FIELD: (UniversalConcept.DEFINITION, "field"),
    ChunkType.VARIABLE: (UniversalConcept.DEFINITION, "variable"),
    ChunkType.BLOCK: (UniversalConcept.BLOCK, None),  # kind varies
    ChunkType.COMMENT: (UniversalConcept.COMMENT, "comment"),
}


def find_by_symbol(chunks, symbol):
    """Filter chunks by symbol/name."""
    return [c for c in chunks if c.name == symbol]


def find_by_type(chunks, chunk_type):
    """Filter chunks by ChunkType via metadata['kind'] or concept.

    Maps ChunkType to the equivalent UniversalConcept + metadata['kind'].
    """
    mapping = _CHUNK_TYPE_MAPPING.get(chunk_type)
    if mapping is None:
        return []
    concept, kind = mapping
    if kind is None:
        # For BLOCK, just filter by concept
        return [c for c in chunks if c.concept == concept]
    return [
        c for c in chunks if c.concept == concept and c.metadata.get("kind") == kind
    ]


def find_by_metadata(chunks, key, value):
    """Filter chunks by metadata key-value pair."""
    return [c for c in chunks if c.metadata and c.metadata.get(key) == value]


def assert_no_parse_errors(parser: TwinCATParser) -> None:
    """Assert that parser has no parse errors. Call after parsing."""
    assert parser.parse_errors == [], (
        f"Parser encountered errors: {parser.parse_errors}"
    )


def extract_chunks_from_file(
    parser: TwinCATParser, file_path: Path
) -> list[UniversalChunk]:
    """Extract UniversalChunks from a TcPOU file."""
    return parser.extract_universal_chunks(file_path.read_text(), file_path)


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def twincat_parser():
    """Create TwinCAT parser instance."""
    return TwinCATParser()


@pytest.fixture
def comprehensive_fixture():
    """Load the comprehensive test fixture."""
    fixture_path = (
        Path(__file__).parent / "fixtures" / "twincat" / "example_comprehensive.TcPOU"
    )
    if not fixture_path.exists():
        pytest.skip("Comprehensive fixture not found")
    return fixture_path


@pytest.fixture
def program_fixture():
    """Load the PROGRAM test fixture."""
    fixture_path = (
        Path(__file__).parent / "fixtures" / "twincat" / "example_program.TcPOU"
    )
    if not fixture_path.exists():
        pytest.skip("PROGRAM fixture not found")
    return fixture_path


@pytest.fixture
def function_fixture():
    """Load the FUNCTION test fixture."""
    fixture_path = (
        Path(__file__).parent / "fixtures" / "twincat" / "example_function.TcPOU"
    )
    if not fixture_path.exists():
        pytest.skip("FUNCTION fixture not found")
    return fixture_path


# =============================================================================
# TestTwinCATParserAvailability
# =============================================================================


class TestTwinCATParserAvailability:
    """Test parser availability and file detection."""

    def test_twincat_parser_instantiation(self):
        """Test that TwinCATParser can be instantiated directly."""
        parser = TwinCATParser()
        assert parser is not None

    def test_twincat_file_detection(self):
        """Test that factory detects .TcPOU extension correctly."""
        factory = ParserFactory()
        detected = factory.detect_language(Path("test.TcPOU"))
        assert detected == Language.TWINCAT

    def test_twincat_file_detection_lowercase(self):
        """Test that factory detects .tcpou extension (lowercase)."""
        factory = ParserFactory()
        detected = factory.detect_language(Path("test.tcpou"))
        assert detected == Language.TWINCAT


# =============================================================================
# TestPOUTypes
# =============================================================================


class TestPOUTypes:
    """Test PROGRAM and FUNCTION POU types."""

    # --- PROGRAM Tests ---

    def test_program_chunk_type(self, twincat_parser, program_fixture):
        """Test PROGRAM creates separate declaration/implementation chunks."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        program_chunks = find_by_type(chunks, ChunkType.PROGRAM)
        # Now 2 chunks: declaration + implementation
        assert len(program_chunks) == 2
        chunk_names = {c.name for c in program_chunks}
        assert chunk_names == {"PRG_Example.declaration", "PRG_Example.implementation"}

    def test_program_metadata(self, twincat_parser, program_fixture):
        """Test PROGRAM metadata includes kind, pou_type, section."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        program_chunks = find_by_type(chunks, ChunkType.PROGRAM)
        assert len(program_chunks) == 2
        # Check declaration chunk
        decl = [c for c in program_chunks if c.name == "PRG_Example.declaration"][0]
        assert decl.metadata["kind"] == "program"
        assert decl.metadata["pou_type"] == "PROGRAM"
        assert decl.metadata["pou_name"] == "PRG_Example"
        assert decl.metadata["pou_id"] == "{11111111-1111-1111-1111-111111111111}"
        assert decl.metadata["section"] == "declaration"

    def test_program_variables(self, twincat_parser, program_fixture):
        """Test PROGRAM extracts VAR_INPUT, VAR_OUTPUT, and VAR blocks."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)

        # VAR_INPUT: bStart
        input_vars = find_by_metadata(chunks, "var_class", "input")
        assert len(input_vars) == 1
        assert input_vars[0].name == "PRG_Example.bStart"
        assert input_vars[0].metadata["data_type"] == "BOOL"

        # VAR_OUTPUT: bRunning
        output_vars = find_by_metadata(chunks, "var_class", "output")
        assert len(output_vars) == 1
        assert output_vars[0].name == "PRG_Example.bRunning"
        assert output_vars[0].metadata["data_type"] == "BOOL"

        # VAR (local): nCycleCount
        local_vars = find_by_metadata(chunks, "var_class", "local")
        assert len(local_vars) == 1
        assert local_vars[0].name == "PRG_Example.nCycleCount"
        assert local_vars[0].metadata["data_type"] == "DINT"

    # --- FUNCTION Tests ---

    def test_function_chunk_type(self, twincat_parser, function_fixture):
        """Test FUNCTION creates separate declaration/implementation chunks."""
        chunks = extract_chunks_from_file(twincat_parser, function_fixture)
        assert_no_parse_errors(twincat_parser)
        function_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        # Now 2 chunks: declaration + implementation
        assert len(function_chunks) == 2
        chunk_names = {c.name for c in function_chunks}
        assert chunk_names == {"FC_Add.declaration", "FC_Add.implementation"}

    def test_function_metadata(self, twincat_parser, function_fixture):
        """Test FUNCTION metadata includes kind, pou_type, section."""
        chunks = extract_chunks_from_file(twincat_parser, function_fixture)
        assert_no_parse_errors(twincat_parser)
        function_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(function_chunks) == 2
        # Check declaration chunk
        decl = [c for c in function_chunks if c.name == "FC_Add.declaration"][0]
        assert decl.metadata["kind"] == "function"
        assert decl.metadata["pou_type"] == "FUNCTION"
        assert decl.metadata["pou_name"] == "FC_Add"
        assert decl.metadata["pou_id"] == "{22222222-2222-2222-2222-222222222222}"
        assert decl.metadata["section"] == "declaration"

    def test_function_variables(self, twincat_parser, function_fixture):
        """Test FUNCTION extracts VAR_INPUT and VAR blocks."""
        chunks = extract_chunks_from_file(twincat_parser, function_fixture)
        assert_no_parse_errors(twincat_parser)

        # VAR_INPUT: nA, nB
        input_vars = find_by_metadata(chunks, "var_class", "input")
        assert len(input_vars) == 2
        input_names = {c.name for c in input_vars}
        assert input_names == {"FC_Add.nA", "FC_Add.nB"}
        for var in input_vars:
            assert var.metadata["data_type"] == "DINT"

        # VAR (local): nResult
        local_vars = find_by_metadata(chunks, "var_class", "local")
        assert len(local_vars) == 1
        assert local_vars[0].name == "FC_Add.nResult"
        assert local_vars[0].metadata["data_type"] == "DINT"


# =============================================================================
# TestPOUChunkCreation
# =============================================================================


class TestPOUChunkCreation:
    """Test POU (Program Organization Unit) chunk creation."""

    def test_function_block_chunk(self, twincat_parser):
        """Test FUNCTION_BLOCK creates separate declaration/implementation chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nValue := 42;]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        pou_chunks = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)
        # Now 2 chunks: declaration + implementation
        assert len(pou_chunks) == 2
        chunk_names = {c.name for c in pou_chunks}
        assert chunk_names == {"FB_Test.declaration", "FB_Test.implementation"}

    def test_pou_metadata(self, twincat_parser):
        """Test POU metadata includes pou_type, pou_name, pou_id, kind, section."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{aaaa-bbbb-cccc-dddd}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        pou_chunks = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)
        # Only declaration (empty implementation)
        assert len(pou_chunks) == 1
        metadata = pou_chunks[0].metadata
        assert metadata["kind"] == "function_block"
        assert metadata["pou_type"] == "FUNCTION_BLOCK"
        assert metadata["pou_name"] == "FB_Test"
        assert metadata["pou_id"] == "{aaaa-bbbb-cccc-dddd}"
        assert metadata["section"] == "declaration"


# =============================================================================
# TestVariableClassification
# =============================================================================


class TestVariableClassification:
    """Test VAR block classification to VARIABLE vs FIELD chunk types."""

    def test_var_input_is_field(self, twincat_parser):
        """Test VAR_INPUT creates FIELD chunk with var_class='input'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_INPUT
    bEnable : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.bEnable")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "input"

    def test_var_output_is_field(self, twincat_parser):
        """Test VAR_OUTPUT creates FIELD chunk with var_class='output'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_OUTPUT
    bDone : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.bDone")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "output"

    def test_var_in_out_is_field(self, twincat_parser):
        """Test VAR_IN_OUT creates FIELD chunk with var_class='in_out'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_IN_OUT
    refData : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.refData")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "in_out"

    def test_var_local_is_field(self, twincat_parser):
        """Test VAR (local) creates FIELD chunk with var_class='local'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nLocal : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.nLocal")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "local"

    def test_var_stat_is_field(self, twincat_parser):
        """Test VAR_STAT creates FIELD chunk with var_class='static'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_STAT
    nCounter : UDINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.nCounter")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "static"

    def test_var_temp_is_field(self, twincat_parser):
        """Test VAR_TEMP creates FIELD chunk with var_class='temp'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_TEMP
    nTemp : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.nTemp")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "field"
        assert var_chunks[0].metadata["var_class"] == "temp"

    def test_var_global_is_variable(self, twincat_parser):
        """Test VAR_GLOBAL creates VARIABLE chunk with var_class='global'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_GLOBAL
    gValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_symbol(chunks, "FB_Test.gValue")
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "variable"
        assert var_chunks[0].metadata["var_class"] == "global"

    def test_var_external_is_variable(self, twincat_parser):
        """Test VAR_EXTERNAL creates VARIABLE chunk with var_class='external'.

        Note: VAR_EXTERNAL also creates an IMPORT chunk, but this test verifies
        the DEFINITION chunk specifically.
        """
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_EXTERNAL
    extValue : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        # Filter by symbol AND concept=DEFINITION (there's also an IMPORT chunk)
        var_chunks = [
            c
            for c in find_by_symbol(chunks, "FB_Test.extValue")
            if c.concept == UniversalConcept.DEFINITION
        ]
        assert len(var_chunks) == 1
        assert var_chunks[0].concept == UniversalConcept.DEFINITION
        assert var_chunks[0].metadata["kind"] == "variable"
        assert var_chunks[0].metadata["var_class"] == "external"


# =============================================================================
# TestDataTypeExtraction
# =============================================================================


class TestDataTypeExtraction:
    """Test data type extraction from variable declarations."""

    def test_primitive_types(self, twincat_parser):
        """Test primitive type extraction (BOOL, INT, DINT, REAL, etc.)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    bFlag : BOOL;
    nInt : INT;
    nDint : DINT;
    fReal : REAL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        bool_chunk = find_by_symbol(chunks, "FB_Test.bFlag")[0]
        assert bool_chunk.metadata["data_type"] == "BOOL"

        int_chunk = find_by_symbol(chunks, "FB_Test.nInt")[0]
        assert int_chunk.metadata["data_type"] == "INT"

        dint_chunk = find_by_symbol(chunks, "FB_Test.nDint")[0]
        assert dint_chunk.metadata["data_type"] == "DINT"

        real_chunk = find_by_symbol(chunks, "FB_Test.fReal")[0]
        assert real_chunk.metadata["data_type"] == "REAL"

    def test_string_with_size(self, twincat_parser):
        """Test STRING(n) and WSTRING(n) type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    sName : STRING(80);
    wsText : WSTRING(100);
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        string_chunk = find_by_symbol(chunks, "FB_Test.sName")[0]
        assert string_chunk.metadata["data_type"] == "STRING(80)"

        wstring_chunk = find_by_symbol(chunks, "FB_Test.wsText")[0]
        assert wstring_chunk.metadata["data_type"] == "WSTRING(100)"

    def test_single_dimension_array(self, twincat_parser):
        """Test ARRAY[0..9] OF INT type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    anData : ARRAY[0..9] OF INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        array_chunk = find_by_symbol(chunks, "FB_Test.anData")[0]
        assert "ARRAY[0..9] OF INT" in array_chunk.metadata["data_type"]

    def test_multi_dimension_array(self, twincat_parser):
        """Test ARRAY[1..3, 1..3] OF REAL type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    afMatrix : ARRAY[1..3, 1..3] OF REAL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        array_chunk = find_by_symbol(chunks, "FB_Test.afMatrix")[0]
        data_type = array_chunk.metadata["data_type"]
        assert "ARRAY" in data_type
        assert "1..3" in data_type
        assert "REAL" in data_type

    def test_pointer_type(self, twincat_parser):
        """Test POINTER TO INT type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    pnValue : POINTER TO INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        ptr_chunk = find_by_symbol(chunks, "FB_Test.pnValue")[0]
        assert "POINTER TO" in ptr_chunk.metadata["data_type"]
        assert "INT" in ptr_chunk.metadata["data_type"]

    def test_reference_type(self, twincat_parser):
        """Test REFERENCE TO DINT type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    refValue : REFERENCE TO DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        ref_chunk = find_by_symbol(chunks, "FB_Test.refValue")[0]
        assert "REFERENCE TO" in ref_chunk.metadata["data_type"]
        assert "DINT" in ref_chunk.metadata["data_type"]

    def test_user_defined_type(self, twincat_parser):
        """Test user-defined types (ST_DataRecord, TON, E_MachineState)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    stData : ST_DataRecord;
    fbTimer : TON;
    eState : E_MachineState;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        struct_chunk = find_by_symbol(chunks, "FB_Test.stData")[0]
        assert struct_chunk.metadata["data_type"] == "ST_DataRecord"

        timer_chunk = find_by_symbol(chunks, "FB_Test.fbTimer")[0]
        assert timer_chunk.metadata["data_type"] == "TON"

        enum_chunk = find_by_symbol(chunks, "FB_Test.eState")[0]
        assert enum_chunk.metadata["data_type"] == "E_MachineState"

    def test_nested_array_type(self, twincat_parser):
        """Test ARRAY[0..1] OF ARRAY[0..2] OF INT type extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    anNestedArray : ARRAY[0..1] OF ARRAY[0..2] OF INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        array_chunk = find_by_symbol(chunks, "FB_Test.anNestedArray")[0]
        data_type = array_chunk.metadata["data_type"]
        assert "ARRAY[0..1]" in data_type
        assert "ARRAY[0..2]" in data_type
        assert "INT" in data_type


# =============================================================================
# TestVariableQualifiers
# =============================================================================


class TestVariableQualifiers:
    """Test RETAIN/PERSISTENT qualifier extraction."""

    def test_retain_qualifier(self, twincat_parser):
        """Test RETAIN qualifier sets metadata['retain'] = True."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR RETAIN
    nRetained : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.nRetained")[0]
        assert var_chunk.metadata["retain"] is True
        assert var_chunk.metadata["persistent"] is False
        assert var_chunk.metadata["constant"] is False

    def test_persistent_qualifier(self, twincat_parser):
        """Test PERSISTENT qualifier sets metadata['persistent'] = True."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR PERSISTENT
    nPersistent : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.nPersistent")[0]
        assert var_chunk.metadata["persistent"] is True
        assert var_chunk.metadata["retain"] is False
        assert var_chunk.metadata["constant"] is False

    def test_retain_persistent_combined(self, twincat_parser):
        """Test RETAIN PERSISTENT sets both qualifiers to True."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR RETAIN PERSISTENT
    stSaved : ST_Data;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.stSaved")[0]
        assert var_chunk.metadata["retain"] is True
        assert var_chunk.metadata["persistent"] is True
        assert var_chunk.metadata["constant"] is False


# =============================================================================
# TestConstantQualifier
# =============================================================================


class TestConstantQualifier:
    """Test CONSTANT qualifier extraction."""

    def test_constant_qualifier(self, twincat_parser):
        """Test VAR CONSTANT sets metadata['constant'] = True."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR CONSTANT
    MAX_SIZE : INT := 100;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.MAX_SIZE")[0]
        assert var_chunk.metadata["constant"] is True
        assert var_chunk.metadata["retain"] is False
        assert var_chunk.metadata["persistent"] is False

    def test_non_constant_has_false(self, twincat_parser):
        """Test regular VAR has metadata['constant'] = False."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.nValue")[0]
        assert var_chunk.metadata["constant"] is False

    def test_var_global_constant_combination(self, twincat_parser):
        """Test VAR_GLOBAL CONSTANT sets both var_class='global' and constant=True."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_GLOBAL CONSTANT
    G_MAX_SIZE : INT := 1000;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.G_MAX_SIZE")[0]
        assert var_chunk.concept == UniversalConcept.DEFINITION
        assert var_chunk.metadata["kind"] == "variable"  # GLOBAL → VARIABLE
        assert var_chunk.metadata["var_class"] == "global"
        assert var_chunk.metadata["constant"] is True


# =============================================================================
# TestHardwareAddressing
# =============================================================================


class TestHardwareAddressing:
    """Test AT directive and hardware address extraction."""

    def test_hw_address_input_wildcard(self, twincat_parser):
        """Test AT %I* extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    bDigitalInput AT %I* : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.bDigitalInput")[0]
        assert var_chunk.metadata["hw_address"] == "%I*"

    def test_hw_address_input_word(self, twincat_parser):
        """Test AT %IW100 extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nAnalogInput AT %IW100 : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.nAnalogInput")[0]
        assert var_chunk.metadata["hw_address"] == "%IW100"

    def test_hw_address_output(self, twincat_parser):
        """Test AT %Q* and %QW50 extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    bDigitalOutput AT %Q* : BOOL;
    nAnalogOutput AT %QW50 : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        digital_chunk = find_by_symbol(chunks, "FB_Test.bDigitalOutput")[0]
        assert digital_chunk.metadata["hw_address"] == "%Q*"

        analog_chunk = find_by_symbol(chunks, "FB_Test.nAnalogOutput")[0]
        assert analog_chunk.metadata["hw_address"] == "%QW50"

    def test_hw_address_memory(self, twincat_parser):
        """Test AT %MW200 and %MX100.0 extraction."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nMemoryWord AT %MW200 : WORD;
    bMemoryBit AT %MX100.0 : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        word_chunk = find_by_symbol(chunks, "FB_Test.nMemoryWord")[0]
        assert word_chunk.metadata["hw_address"] == "%MW200"

        bit_chunk = find_by_symbol(chunks, "FB_Test.bMemoryBit")[0]
        assert bit_chunk.metadata["hw_address"] == "%MX100.0"


# =============================================================================
# TestMultipleVariablesDeclaration
# =============================================================================


class TestMultipleVariablesDeclaration:
    """Test comma-separated variable declarations."""

    def test_comma_separated_variables(self, twincat_parser):
        """Test that bFlag1, bFlag2, bFlag3 : BOOL; creates 3 separate chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    bFlag1, bFlag2, bFlag3 : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        flag_chunks = [
            c
            for c in chunks
            if c.name in ("FB_Test.bFlag1", "FB_Test.bFlag2", "FB_Test.bFlag3")
        ]
        assert len(flag_chunks) == 3

        # All should have BOOL data type
        for chunk in flag_chunks:
            assert chunk.metadata["data_type"] == "BOOL"
            assert chunk.concept == UniversalConcept.DEFINITION
            assert chunk.metadata["kind"] == "field"


# =============================================================================
# TestActionChunks
# =============================================================================


class TestActionChunks:
    """Test ACTION chunk creation and metadata."""

    def test_action_chunk_created(self, twincat_parser):
        """Test ACTION creates separate declaration/implementation chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="ProcessData" Id="{action-id-1}">
      <Declaration><![CDATA[VAR
    nLocal : INT;
END_VAR
]]></Declaration>
      <Implementation><ST><![CDATA[nValue := 42;]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_chunks = find_by_type(chunks, ChunkType.ACTION)
        # 2 chunks: declaration + implementation
        assert len(action_chunks) == 2
        chunk_names = {c.name for c in action_chunks}
        assert "FB_Test.ProcessData.declaration" in chunk_names
        assert "FB_Test.ProcessData.implementation" in chunk_names

    def test_action_metadata(self, twincat_parser):
        """Test action metadata includes kind='action', pou_name, action_id, section."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="MyAction" Id="{action-uuid}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation><ST><![CDATA[nValue := 1;]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_chunks = find_by_type(chunks, ChunkType.ACTION)
        # Only implementation (empty declaration)
        assert len(action_chunks) == 1
        metadata = action_chunks[0].metadata
        assert metadata["kind"] == "action"
        assert metadata["pou_name"] == "FB_Test"
        assert metadata["action_id"] == "{action-uuid}"
        assert metadata["section"] == "implementation"

    def test_action_local_variables_have_action_name(self, twincat_parser):
        """Test action local variables have metadata['action_name'] set."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="ProcessData" Id="{action-id}">
      <Declaration><![CDATA[VAR
    nLocalIndex : INT;
    fLocalSum : REAL;
END_VAR
]]></Declaration>
      <Implementation><ST><![CDATA[]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_var_chunks = find_by_metadata(chunks, "action_name", "ProcessData")
        assert len(action_var_chunks) == 2

        for chunk in action_var_chunks:
            assert chunk.metadata["action_name"] == "ProcessData"

    def test_action_local_variables_are_fields(self, twincat_parser):
        """Test action local variables are ChunkType.FIELD, not VARIABLE."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="TestAction" Id="{action-id}">
      <Declaration><![CDATA[VAR
    nActionLocal : INT;
END_VAR
]]></Declaration>
      <Implementation><ST><![CDATA[]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_var = find_by_symbol(chunks, "FB_Test.TestAction.nActionLocal")[0]
        assert action_var.concept == UniversalConcept.DEFINITION
        assert action_var.metadata["kind"] == "field"
        assert action_var.metadata["var_class"] == "local"

    def test_multiple_actions(self, twincat_parser):
        """Test multiple actions create separate implementation chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="ActionOne" Id="{action-1}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation><ST><![CDATA[nValue := 1;]]></ST></Implementation>
    </Action>
    <Action Name="ActionTwo" Id="{action-2}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation><ST><![CDATA[nValue := 2;]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_chunks = find_by_type(chunks, ChunkType.ACTION)
        # 2 implementation chunks (empty declarations skipped)
        assert len(action_chunks) == 2

        action_names = {c.name for c in action_chunks}
        assert action_names == {
            "FB_Test.ActionOne.implementation",
            "FB_Test.ActionTwo.implementation",
        }


# =============================================================================
# TestComprehensiveFixture
# =============================================================================


class TestComprehensiveFixture:
    """Integration tests using the comprehensive fixture file."""

    def test_fixture_parses_without_errors(self, twincat_parser, comprehensive_fixture):
        """Test fixture parses without parse_errors."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert len(chunks) > 0
        assert len(twincat_parser.parse_errors) == 0

    def test_fixture_main_pou(self, twincat_parser, comprehensive_fixture):
        """Test fixture has FB_ComprehensiveExample FUNCTION_BLOCK chunks."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        pou_chunks = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)
        # Now 2 chunks: declaration + implementation
        assert len(pou_chunks) == 2
        chunk_names = {c.name for c in pou_chunks}
        assert "FB_ComprehensiveExample.declaration" in chunk_names
        assert "FB_ComprehensiveExample.implementation" in chunk_names

    def test_fixture_var_input_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts all VAR_INPUT variables."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        input_vars = find_by_metadata(chunks, "var_class", "input")
        # Fixture has: bEnable, nInputValue, fSetpoint, sCommand, anInputArray, afMatrix
        assert len(input_vars) >= 6

    def test_fixture_var_output_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts all VAR_OUTPUT variables."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        output_vars = find_by_metadata(chunks, "var_class", "output")
        # From fixture: bDone, bError, nErrorCode, sStatus, anResults
        assert len(output_vars) >= 5

    def test_fixture_var_in_out_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts both VAR_IN_OUT variables."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        in_out_vars = find_by_metadata(chunks, "var_class", "in_out")
        # From fixture: refCounter, aBuffer
        assert len(in_out_vars) >= 2

    def test_fixture_var_stat_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts VAR_STAT variables with var_class='static'."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        static_vars = find_by_metadata(chunks, "var_class", "static")
        # From fixture: nCallCount, fAccumulator
        assert len(static_vars) >= 2

    def test_fixture_var_temp_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts VAR_TEMP variables with var_class='temp'."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        temp_vars = find_by_metadata(chunks, "var_class", "temp")
        # From fixture: nTempValue, fTempResult
        assert len(temp_vars) >= 2

    def test_fixture_retain_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts RETAIN variables with retain=True."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        retain_vars = find_by_metadata(chunks, "retain", True)
        # From fixture: nRetainedCounter, bRetainedFlag, stSavedState
        assert len(retain_vars) >= 2

    def test_fixture_persistent_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts PERSISTENT variables with persistent=True."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        persistent_vars = find_by_metadata(chunks, "persistent", True)
        # From fixture: nPersistentValue, stSavedState
        assert len(persistent_vars) >= 1

    def test_fixture_hardware_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts AT directive variables with hw_address."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        hw_vars = [
            c for c in chunks if c.metadata and c.metadata.get("hw_address") is not None
        ]
        # From fixture: bDigitalInput, nAnalogInput, bDigitalOutput, nAnalogOutput,
        #               nMemoryWord, bMemoryBit
        assert len(hw_vars) >= 6

    def test_fixture_actions(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts ProcessData and ResetState ACTION chunks."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        action_chunks = find_by_type(chunks, ChunkType.ACTION)
        # Each action has declaration + implementation = 4 total
        assert len(action_chunks) == 4

        # Check implementation chunks exist
        impl_names = {
            c.name
            for c in action_chunks
            if c.metadata.get("section") == "implementation"
        }
        assert "FB_ComprehensiveExample.ProcessData.implementation" in impl_names
        assert "FB_ComprehensiveExample.ResetState.implementation" in impl_names

    def test_fixture_action_local_vars(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts action-scoped variables with action_name."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)

        # ProcessData action vars: nLocalIndex, fLocalSum, bLocalFlag, anLocalBuffer
        process_data_vars = find_by_metadata(chunks, "action_name", "ProcessData")
        assert len(process_data_vars) >= 4

        # ResetState action var: nResetIndex
        reset_state_vars = find_by_metadata(chunks, "action_name", "ResetState")
        assert len(reset_state_vars) >= 1

    def test_fixture_constant_variables(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts VAR CONSTANT variables with constant=True."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        constant_vars = find_by_metadata(chunks, "constant", True)
        # From fixture: c_nMaxItems, c_fPi, c_sVersion
        assert len(constant_vars) >= 3

    def test_fixture_control_flow_blocks(self, twincat_parser, comprehensive_fixture):
        """Test fixture extracts BLOCK chunks for control flow in implementation."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        # Fixture has: multiple IF, CASE, FOR, WHILE, REPEAT blocks
        assert len(block_chunks) >= 5

        # Verify all control flow types are represented
        block_kinds = {b.metadata["kind"] for b in block_chunks}
        assert "if_block" in block_kinds
        assert "case_block" in block_kinds
        assert "for_loop" in block_kinds
        assert "while_loop" in block_kinds
        assert "repeat_loop" in block_kinds

    def test_fixture_all_chunks_have_language_node_type(
        self, twincat_parser, comprehensive_fixture
    ):
        """Test all chunks have language_node_type set (indicates TwinCAT/Lark origin)."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        for chunk in chunks:
            # UniversalChunk is language-agnostic but tracks origin via language_node_type
            assert chunk.language_node_type is not None
            assert chunk.language_node_type.startswith("lark_")


# =============================================================================
# TestMetadataCompleteness
# =============================================================================


class TestMetadataCompleteness:
    """Test that all required metadata fields are present."""

    def test_pou_metadata_fields(self, twincat_parser):
        """Test POU chunks have all required metadata fields."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{uuid-here}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        pou_chunk = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)[0]

        required_fields = ["kind", "pou_type", "pou_name", "pou_id"]
        for field in required_fields:
            assert field in pou_chunk.metadata, f"Missing field: {field}"

    def test_variable_metadata_fields(self, twincat_parser):
        """Test variable chunks have all expected metadata fields."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR RETAIN
    nValue AT %MW100 : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        var_chunk = find_by_symbol(chunks, "FB_Test.nValue")[0]

        expected_fields = [
            "kind",
            "pou_type",
            "pou_name",
            "var_class",
            "data_type",
            "hw_address",
            "retain",
            "persistent",
        ]
        for field in expected_fields:
            assert field in var_chunk.metadata, f"Missing field: {field}"

    def test_action_metadata_fields(self, twincat_parser):
        """Test action chunks have all required metadata fields."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="MyAction" Id="{action-uuid}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation><ST><![CDATA[n := 1;]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        action_chunk = find_by_type(chunks, ChunkType.ACTION)[0]

        required_fields = ["kind", "pou_type", "pou_name", "action_id"]
        for field in required_fields:
            assert field in action_chunk.metadata, f"Missing field: {field}"


# =============================================================================
# TestParseErrorHandling
# =============================================================================


# =============================================================================
# TestTwinCATLineNumbers
# =============================================================================


class TestTwinCATLineNumbers:
    """Test that line numbers are absolute (relative to XML file, not CDATA)."""

    def test_pou_chunk_has_absolute_start_line(self, twincat_parser, program_fixture):
        """Test POU declaration chunk start_line equals CDATA start (line 4)."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        pou_chunks = find_by_type(chunks, ChunkType.PROGRAM)
        assert len(pou_chunks) == 2
        # Declaration CDATA starts at line 4 in example_program.TcPOU
        decl_chunk = [
            c for c in pou_chunks if c.metadata.get("section") == "declaration"
        ][0]
        assert decl_chunk.start_line == 4

    def test_pou_chunk_end_line_spans_content(self, twincat_parser, program_fixture):
        """Test POU implementation end_line (line 20)."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        pou_chunks = find_by_type(chunks, ChunkType.PROGRAM)
        assert len(pou_chunks) == 2
        # Implementation CDATA content spans lines 16-20
        impl_chunk = [
            c for c in pou_chunks if c.metadata.get("section") == "implementation"
        ][0]
        assert impl_chunk.end_line == 20

    def test_variable_chunks_have_absolute_line_numbers(
        self, twincat_parser, program_fixture
    ):
        """Test that variable line numbers are > 3 (not CDATA-relative starting at 1)."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        var_chunks = find_by_type(chunks, ChunkType.FIELD)
        assert len(var_chunks) >= 3  # bStart, bRunning, nCycleCount

        for chunk in var_chunks:
            # CDATA-relative would start at 1; absolute must be > 3
            assert chunk.start_line > 3, (
                f"Variable {chunk.name} has line {chunk.start_line}, "
                "which appears to be CDATA-relative, not XML-absolute"
            )

    def test_variable_line_numbers_match_xml_positions(
        self, twincat_parser, program_fixture
    ):
        """Test specific variable line numbers: bStart=6, bRunning=9, nCycleCount=12."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)

        # bStart is on line 6
        bstart_chunks = find_by_symbol(chunks, "PRG_Example.bStart")
        assert len(bstart_chunks) == 1
        assert bstart_chunks[0].start_line == 6

        # bRunning is on line 9
        brunning_chunks = find_by_symbol(chunks, "PRG_Example.bRunning")
        assert len(brunning_chunks) == 1
        assert brunning_chunks[0].start_line == 9

        # nCycleCount is on line 12
        ncycle_chunks = find_by_symbol(chunks, "PRG_Example.nCycleCount")
        assert len(ncycle_chunks) == 1
        assert ncycle_chunks[0].start_line == 12

    def test_line_numbers_are_not_cdata_relative(self, twincat_parser, program_fixture):
        """Verify line numbers are > 3 (XML-absolute, not CDATA-relative starting at 1)."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)

        for chunk in chunks:
            # All chunks should have line numbers > 3 because:
            # - Line 1: XML declaration
            # - Line 2: TcPlcObject
            # - Line 3: POU element
            # - Line 4+: actual content in CDATA
            assert chunk.start_line > 3, (
                f"Chunk {chunk.name} has start_line={chunk.start_line}, "
                "which suggests CDATA-relative numbering"
            )
            assert chunk.end_line >= chunk.start_line, (
                f"Chunk {chunk.name} has invalid end_line={chunk.end_line} "
                f"< start_line={chunk.start_line}"
            )


# =============================================================================
# TestParseErrorHandling
# =============================================================================


class TestParseErrorHandling:
    """Test that parse_errors is properly populated on grammar errors."""

    def test_invalid_declaration_populates_parse_errors(self, twincat_parser):
        """Test that invalid declaration syntax populates parse_errors."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    THIS IS INVALID SYNTAX !!!
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        # Should still create the POU chunk (graceful degradation)
        pou_chunks = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)
        assert len(pou_chunks) == 1
        # But parse_errors should be populated
        assert len(twincat_parser.parse_errors) > 0
        assert "parse error" in twincat_parser.parse_errors[0].lower()

    def test_invalid_action_declaration_populates_parse_errors(self, twincat_parser):
        """Test that invalid action declaration populates parse_errors."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Action Name="BadAction" Id="{action-id}">
      <Declaration><![CDATA[VAR
    INVALID DECLARATION GARBAGE @#$%
END_VAR
]]></Declaration>
      <Implementation><ST><![CDATA[]]></ST></Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        # Main POU and action should still be created
        pou_chunks = find_by_type(chunks, ChunkType.FUNCTION_BLOCK)
        action_chunks = find_by_type(chunks, ChunkType.ACTION)
        assert len(pou_chunks) == 1
        assert len(action_chunks) == 1
        # But parse_errors should be populated with action error
        assert len(twincat_parser.parse_errors) > 0
        assert "BadAction" in twincat_parser.parse_errors[0]

    def test_parse_errors_cleared_between_parses(self, twincat_parser):
        """Test that parse_errors is cleared between parse operations."""
        # First parse with invalid syntax
        invalid_xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    INVALID SYNTAX @#$
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        twincat_parser.extract_universal_chunks(invalid_xml)
        assert len(twincat_parser.parse_errors) > 0

        # Second parse with valid syntax
        valid_xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        twincat_parser.extract_universal_chunks(valid_xml)
        # parse_errors should now be empty
        assert len(twincat_parser.parse_errors) == 0


# =============================================================================
# TestImplementationBlockExtraction
# =============================================================================


class TestImplementationBlockExtraction:
    """Test control flow BLOCK chunk extraction from FUNCTION implementations."""

    # --- Pytest Fixtures ---

    @pytest.fixture
    def function_if_fixture(self):
        """Load the FUNCTION IF test fixture."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "twincat" / "example_function_if.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION IF fixture not found")
        return fixture_path

    @pytest.fixture
    def function_loops_fixture(self):
        """Load the FUNCTION loops test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_loops.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION loops fixture not found")
        return fixture_path

    @pytest.fixture
    def function_action_fixture(self):
        """Load the FUNCTION with action test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_with_action.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION with action fixture not found")
        return fixture_path

    # --- POU Implementation Tests ---

    def test_function_if_block(self, twincat_parser):
        """Test that FUNCTION with IF creates BLOCK chunk with kind='if_block'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF n > 0 THEN
    FC_Test := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) >= 1
        if_blocks = find_by_metadata(block_chunks, "kind", "if_block")
        assert len(if_blocks) == 1
        assert if_blocks[0].metadata["pou_type"] == "FUNCTION"
        assert if_blocks[0].metadata["pou_name"] == "FC_Test"
        # Verify FQN symbol pattern: POUName.{kind}_{line}
        assert if_blocks[0].name.startswith("FC_Test.if_block_")

    def test_function_case_block(self, twincat_parser):
        """Test that FUNCTION with CASE creates BLOCK chunk with kind='case_block'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[CASE n OF
    1: FC_Test := 10;
    2: FC_Test := 20;
ELSE
    FC_Test := 0;
END_CASE;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        case_blocks = find_by_metadata(block_chunks, "kind", "case_block")
        assert len(case_blocks) == 1

    def test_function_for_loop(self, twincat_parser):
        """Test that FUNCTION with FOR creates BLOCK chunk with kind='for_loop'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    i : INT;
    sum : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[sum := 0;
FOR i := 0 TO 10 DO
    sum := sum + i;
END_FOR;
FC_Test := sum;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        for_loops = find_by_metadata(block_chunks, "kind", "for_loop")
        assert len(for_loops) == 1

    def test_function_while_loop(self, twincat_parser):
        """Test that FUNCTION with WHILE creates BLOCK chunk with kind='while_loop'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 0;
WHILE i < 10 DO
    i := i + 1;
END_WHILE;
FC_Test := i;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        while_loops = find_by_metadata(block_chunks, "kind", "while_loop")
        assert len(while_loops) == 1

    def test_function_repeat_loop(self, twincat_parser):
        """Test that FUNCTION with REPEAT creates BLOCK chunk with kind='repeat_loop'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 0;
REPEAT
    i := i + 1;
UNTIL i >= 10
END_REPEAT;
FC_Test := i;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        repeat_loops = find_by_metadata(block_chunks, "kind", "repeat_loop")
        assert len(repeat_loops) == 1

    def test_function_multiple_blocks(self, twincat_parser, function_loops_fixture):
        """Test that multiple blocks are extracted from a single FUNCTION."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        # Fixture has: 1 CASE with nested (FOR, WHILE, REPEAT) = 4 total
        assert len(block_chunks) >= 4

    def test_function_nested_blocks(self, twincat_parser, function_if_fixture):
        """Test that nested IF blocks are both extracted."""
        chunks = extract_chunks_from_file(twincat_parser, function_if_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        if_blocks = find_by_metadata(block_chunks, "kind", "if_block")
        # Fixture has outer IF and nested IF
        assert len(if_blocks) == 2

    def test_function_block_metadata(self, twincat_parser):
        """Test BLOCK chunks have correct metadata fields."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MetadataTest" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MetadataTest : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF n > 0 THEN
    FC_MetadataTest := n;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        metadata = block_chunks[0].metadata
        assert metadata["kind"] == "if_block"
        assert metadata["pou_type"] == "FUNCTION"
        assert metadata["pou_name"] == "FC_MetadataTest"
        assert "action_name" not in metadata

    def test_function_block_line_numbers(self, twincat_parser):
        """Test BLOCK chunks have XML-absolute line numbers."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 1;
IF n > 0 THEN
    FC_Test := n;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        # IF starts at line 2 within CDATA, CDATA starts at line 10
        # So absolute line should be > 10
        assert block_chunks[0].start_line > 10
        assert block_chunks[0].end_line >= block_chunks[0].start_line

    # --- Action Implementation Tests ---

    def test_function_action_if_block(self, twincat_parser):
        """Test Action within FUNCTION creates BLOCK with action_name in metadata."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[FC_Test := 0;]]></ST></Implementation>
    <Action Name="DoProcess" Id="{action-1}">
      <Declaration><![CDATA[VAR
    i : INT;
END_VAR
]]></Declaration>
      <Implementation>
        <ST><![CDATA[IF n > 0 THEN
    n := n - 1;
END_IF;
]]></ST>
      </Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        assert block_chunks[0].metadata["kind"] == "if_block"
        assert block_chunks[0].metadata["action_name"] == "DoProcess"
        assert block_chunks[0].metadata["pou_name"] == "FC_Test"
        # Verify FQN symbol pattern: POUName.ActionName.{kind}_{line}
        assert block_chunks[0].name.startswith("FC_Test.DoProcess.if_block_")

    def test_function_action_multiple_blocks(
        self, twincat_parser, function_action_fixture
    ):
        """Test Action with multiple control flow blocks extracts all of them."""
        chunks = extract_chunks_from_file(twincat_parser, function_action_fixture)
        assert_no_parse_errors(twincat_parser)

        # Get action blocks only (those with action_name in metadata)
        action_blocks = [
            c
            for c in chunks
            if c.concept == UniversalConcept.BLOCK and c.metadata.get("action_name")
        ]
        # Action has: FOR (with nested IF), WHILE = 3 blocks
        assert len(action_blocks) >= 3

    def test_function_action_block_metadata(self, twincat_parser):
        """Test action BLOCK chunks include action_name in metadata."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[FC_Test := 0;]]></ST></Implementation>
    <Action Name="MyAction" Id="{action-uuid}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation>
        <ST><![CDATA[FOR n := 0 TO 5 DO
    FC_Test := FC_Test + n;
END_FOR;
]]></ST>
      </Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        metadata = block_chunks[0].metadata
        assert metadata["kind"] == "for_loop"
        assert metadata["action_name"] == "MyAction"
        assert metadata["pou_type"] == "FUNCTION"
        assert metadata["pou_name"] == "FC_Test"

    # --- PROGRAM and FUNCTION_BLOCK Block Extraction Tests ---

    def test_program_block_extraction(self, twincat_parser, program_fixture):
        """Test that PROGRAM extracts BLOCK chunks for control flow."""
        chunks = extract_chunks_from_file(twincat_parser, program_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        # program_fixture contains IF/CASE/FOR/WHILE/REPEAT
        assert len(block_chunks) >= 1

    def test_function_block_block_extraction(self, twincat_parser):
        """Test that FUNCTION_BLOCK extracts BLOCK chunks for control flow."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF n > 0 THEN
    n := n - 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        block = block_chunks[0]
        assert block.metadata["kind"] == "if_block"
        assert block.metadata["pou_type"] == "FUNCTION_BLOCK"
        assert block.metadata["pou_name"] == "FB_Test"

    def test_program_action_block_extraction(self, twincat_parser):
        """Test that PROGRAM actions extract BLOCK chunks for control flow."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="PRG_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[PROGRAM PRG_Test
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[n := 1;]]></ST></Implementation>
    <Action Name="ProgramAction" Id="{action-1}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation>
        <ST><![CDATA[IF n > 0 THEN
    n := 0;
END_IF;
]]></ST>
      </Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        block = block_chunks[0]
        assert block.metadata["kind"] == "if_block"
        assert block.metadata["pou_type"] == "PROGRAM"
        assert block.metadata["action_name"] == "ProgramAction"

    def test_function_block_multiple_blocks(self, twincat_parser):
        """Test FUNCTION_BLOCK with multiple control flow blocks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Multi" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Multi
VAR
    n : INT;
    state : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF n > 0 THEN
    n := n - 1;
END_IF;

CASE state OF
1: n := 10;
2: n := 20;
END_CASE;

FOR n := 0 TO 10 DO
    state := state + 1;
END_FOR;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 3
        block_kinds = {b.metadata["kind"] for b in block_chunks}
        assert block_kinds == {"if_block", "case_block", "for_loop"}

    def test_function_block_action_block_extraction(self, twincat_parser):
        """Test that FUNCTION_BLOCK actions extract BLOCK chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_WithAction" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_WithAction
VAR
    counter : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[counter := 0;]]></ST></Implementation>
    <Action Name="DoLoop" Id="{action-1}">
      <Declaration><![CDATA[]]></Declaration>
      <Implementation>
        <ST><![CDATA[WHILE counter < 10 DO
    counter := counter + 1;
END_WHILE;
]]></ST>
      </Implementation>
    </Action>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        block = block_chunks[0]
        assert block.metadata["kind"] == "while_loop"
        assert block.metadata["pou_type"] == "FUNCTION_BLOCK"
        assert block.metadata["action_name"] == "DoLoop"

    # --- Edge Case Tests ---

    def test_function_empty_implementation(self, twincat_parser):
        """Test FUNCTION with empty implementation returns no BLOCK chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Empty" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Empty : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 0

    def test_function_no_control_flow(self, twincat_parser):
        """Test FUNCTION without control flow statements returns no BLOCK chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Simple" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Simple : INT
VAR_INPUT
    a : INT;
    b : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[FC_Simple := a + b;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 0

    def test_function_parse_error_handling(self, twincat_parser):
        """Test implementation parse error is logged but doesn't crash."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_BadImpl" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_BadImpl : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF INVALID SYNTAX @#$ THEN
    n := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        # Should not raise an exception
        chunks = twincat_parser.extract_universal_chunks(xml)
        # Should still create the FUNCTION chunk
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation
        # Should have parse error logged
        assert len(twincat_parser.parse_errors) > 0
        assert "parse error" in twincat_parser.parse_errors[0].lower()


# =============================================================================
# TestImplementationNumericLiterals
# =============================================================================


class TestImplementationNumericLiterals:
    """Verify numeric literal formats parse correctly in implementation expressions."""

    def test_hex_literals_in_implementation(self, twincat_parser):
        """Test 16#FF, 16#ABCD hex literals in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_HexLiterals" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_HexLiterals : DINT
VAR
    nByte : BYTE;
    nWord : WORD;
    nDword : DWORD;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nByte := 16#FF;
nWord := 16#ABCD;
nDword := 16#12345678;
IF nByte > 16#80 THEN
    FC_HexLiterals := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_binary_literals_in_implementation(self, twincat_parser):
        """Test 2#10101010 binary literals in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_BinaryLiterals" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_BinaryLiterals : BYTE
VAR
    nByte : BYTE;
    nMask : BYTE;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nByte := 2#10101010;
nMask := 2#11110000;
IF (nByte AND nMask) = 2#10100000 THEN
    FC_BinaryLiterals := nByte;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_octal_literals_in_implementation(self, twincat_parser):
        """Test 8#755 octal literals in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_OctalLiterals" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_OctalLiterals : INT
VAR
    nPermissions : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nPermissions := 8#755;
IF nPermissions > 8#700 THEN
    FC_OctalLiterals := nPermissions;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_scientific_notation_in_implementation(self, twincat_parser):
        """Test 1.5e-3, 2E10 scientific notation in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ScientificNotation" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ScientificNotation : LREAL
VAR
    fSmall : LREAL;
    fLarge : LREAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fSmall := 1.5e-3;
fLarge := 2E10;
IF fSmall < 1.0E-2 THEN
    FC_ScientificNotation := fSmall * fLarge;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_negative_numbers_in_implementation(self, twincat_parser):
        """Test -42, -3.14 negative numbers in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_NegativeNumbers" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_NegativeNumbers : REAL
VAR
    nInt : INT;
    fReal : REAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nInt := -42;
fReal := -3.14;
IF nInt < -10 THEN
    FC_NegativeNumbers := fReal;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationStringAndTimeLiterals
# =============================================================================


class TestImplementationStringAndTimeLiterals:
    """Verify string and time literals parse correctly."""

    def test_single_quoted_strings_in_implementation(self, twincat_parser):
        """Test 'Hello' single-quoted strings in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_SingleQuotedStrings" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_SingleQuotedStrings : INT
VAR
    sMessage : STRING(80);
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[sMessage := 'Hello World';
IF sMessage = 'Hello World' THEN
    FC_SingleQuotedStrings := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_double_quoted_strings_in_implementation(self, twincat_parser):
        """Test "World" double-quoted strings in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_DoubleQuotedStrings" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_DoubleQuotedStrings : INT
VAR
    wsMessage : WSTRING(80);
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[wsMessage := "Hello Wide World";
IF wsMessage = "Hello Wide World" THEN
    FC_DoubleQuotedStrings := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_time_literals_in_implementation(self, twincat_parser):
        """Test T#100ms, T#5s, T#-10s time literals."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_TimeLiterals" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_TimeLiterals : INT
VAR
    tDelay : TIME;
    tInterval : TIME;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[tDelay := T#100ms;
tInterval := T#5s;
IF tDelay < T#1s THEN
    FC_TimeLiterals := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_time_literals_in_function_calls(self, twincat_parser):
        """Test time literals in function block calls like fbTimer(PT := T#5s)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_TimeFunctionCalls" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_TimeFunctionCalls : BOOL
VAR
    fbTON : TON;
    bDone : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fbTON(IN := TRUE, PT := T#5s);
bDone := fbTON.Q;
IF bDone THEN
    FC_TimeFunctionCalls := TRUE;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationTypeConversions
# =============================================================================


class TestImplementationTypeConversions:
    """Verify type conversion functions parse correctly."""

    def test_int_to_real_conversion(self, twincat_parser):
        """Test INT_TO_REAL(nInt) in expressions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_IntToReal" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_IntToReal : REAL
VAR
    nInt : INT;
    fReal : REAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nInt := 42;
fReal := INT_TO_REAL(nInt);
FC_IntToReal := fReal;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_real_to_int_conversion(self, twincat_parser):
        """Test REAL_TO_INT(fReal) in expressions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_RealToInt" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_RealToInt : INT
VAR
    fReal : REAL;
    nInt : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fReal := 3.14;
nInt := REAL_TO_INT(fReal);
FC_RealToInt := nInt;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_dint_to_string_conversion(self, twincat_parser):
        """Test DINT_TO_STRING(nDint) in expressions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_DintToString" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_DintToString : INT
VAR
    nDint : DINT;
    sResult : STRING(80);
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nDint := 123456;
sResult := DINT_TO_STRING(nDint);
FC_DintToString := 1;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_type_conversion_in_condition(self, twincat_parser):
        """Test IF INT_TO_REAL(n) > 0.0 THEN type conversion in conditions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ConversionInCondition" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ConversionInCondition : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 42;
IF INT_TO_REAL(n) > 0.0 THEN
    FC_ConversionInCondition := 1;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_nested_type_conversions(self, twincat_parser):
        """Test INT_TO_REAL(REAL_TO_INT(f)) nested conversions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_NestedConversions" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_NestedConversions : REAL
VAR
    fInput : REAL;
    fRounded : REAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fInput := 3.7;
fRounded := INT_TO_REAL(REAL_TO_INT(fInput));
FC_NestedConversions := fRounded;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationAccessPatterns
# =============================================================================


class TestImplementationAccessPatterns:
    """Verify array, member, and bit access patterns parse correctly."""

    def test_single_dim_array_access(self, twincat_parser):
        """Test arr[i] in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ArrayAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ArrayAccess : INT
VAR
    arr : ARRAY[0..9] OF INT;
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 5;
arr[i] := 42;
FC_ArrayAccess := arr[0] + arr[i];
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_multi_dim_array_access(self, twincat_parser):
        """Test matrix[i, j] in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MultiDimArrayAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MultiDimArrayAccess : INT
VAR
    matrix : ARRAY[0..2, 0..2] OF INT;
    i, j : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 1;
j := 2;
matrix[i, j] := 42;
FC_MultiDimArrayAccess := matrix[0, 0] + matrix[i, j];
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_array_access_in_condition(self, twincat_parser):
        """Test IF arr[0] > 0 THEN array access in conditions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ArrayCondition" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ArrayCondition : INT
VAR
    arr : ARRAY[0..9] OF INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[arr[0] := 10;
IF arr[0] > 0 THEN
    FC_ArrayCondition := arr[0];
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_member_access(self, twincat_parser):
        """Test stData.nValue in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MemberAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MemberAccess : INT
VAR
    stData : ST_Data;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[stData.nValue := 42;
FC_MemberAccess := stData.nValue;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_nested_member_access(self, twincat_parser):
        """Test stData.stNested.fValue in assignments."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_NestedMemberAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_NestedMemberAccess : REAL
VAR
    stData : ST_Outer;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[stData.stNested.fValue := 3.14;
FC_NestedMemberAccess := stData.stNested.fValue;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_bit_access(self, twincat_parser):
        """Test nWord.0, nDword.15 bit access."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_BitAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_BitAccess : BOOL
VAR
    nWord : WORD;
    nDword : DWORD;
    bBit0 : BOOL;
    bBit15 : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nWord := 16#FF01;
nDword := 16#00008000;
bBit0 := nWord.0;
bBit15 := nDword.15;
IF nWord.0 THEN
    FC_BitAccess := TRUE;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_combined_access_patterns(self, twincat_parser):
        """Test stData.anArray[i].nFlags.3 combined access patterns."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_CombinedAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_CombinedAccess : BOOL
VAR
    stData : ST_Complex;
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 2;
stData.anArray[i].nFlags := 16#08;
IF stData.anArray[i].nFlags.3 THEN
    FC_CombinedAccess := TRUE;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationComplexExpressions
# =============================================================================


class TestImplementationComplexExpressions:
    """Verify complex expressions and operator precedence."""

    def test_operator_precedence_expt(self, twincat_parser):
        """Test -2.0 EXPT 2.0 equals -4 (negation binds looser)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ExptPrecedence" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ExptPrecedence : LREAL
VAR
    fResult : LREAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fResult := -2.0 EXPT 2.0;
FC_ExptPrecedence := fResult;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_complex_arithmetic_expression(self, twincat_parser):
        """Test (10 + 5) * 3 - 2 complex arithmetic."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ComplexArithmetic" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ComplexArithmetic : INT
VAR
    nResult : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nResult := (10 + 5) * 3 - 2;
FC_ComplexArithmetic := nResult;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_compound_boolean_conditions(self, twincat_parser):
        """Test (n > 0 AND n < 100) OR bFlag compound conditions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_CompoundBoolean" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_CompoundBoolean : BOOL
VAR
    n : INT;
    bFlag : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 50;
bFlag := FALSE;
IF (n > 0 AND n < 100) OR bFlag THEN
    FC_CompoundBoolean := TRUE;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_short_circuit_operators(self, twincat_parser):
        """Test AND_THEN, OR_ELSE short-circuit operators in conditions."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ShortCircuit" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ShortCircuit : BOOL
VAR
    bCondition1 : BOOL;
    bCondition2 : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[bCondition1 := TRUE;
bCondition2 := FALSE;
IF bCondition1 AND_THEN bCondition2 THEN
    FC_ShortCircuit := TRUE;
ELSIF bCondition1 OR_ELSE bCondition2 THEN
    FC_ShortCircuit := TRUE;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_expt_as_function_call(self, twincat_parser):
        """Test EXPT(2.0, 8.0) function call syntax."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ExptFunction" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ExptFunction : LREAL
VAR
    fResult : LREAL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fResult := EXPT(2.0, 8.0);
FC_ExptFunction := fResult;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationFunctionCalls
# =============================================================================


class TestImplementationFunctionCalls:
    """Verify function/FB call syntaxes parse correctly."""

    def test_function_positional_args(self, twincat_parser):
        """Test MAX(a, b), ABS(-42) function calls with positional args."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_PositionalArgs" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_PositionalArgs : INT
VAR
    a, b : INT;
    nMax : INT;
    nAbs : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[a := 10;
b := 20;
nMax := MAX(a, b);
nAbs := ABS(-42);
FC_PositionalArgs := nMax + nAbs;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_function_named_args(self, twincat_parser):
        """Test fb(IN := bEnable, PT := T#5s) named argument calls."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_NamedArgs" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_NamedArgs : BOOL
VAR
    fbTON : TON;
    bEnable : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[bEnable := TRUE;
fbTON(IN := bEnable, PT := T#5s);
FC_NamedArgs := fbTON.Q;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_function_output_args(self, twincat_parser):
        """Test fb(Q => bDone, ET => tDuration) output argument calls."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_OutputArgs" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_OutputArgs : BOOL
VAR
    fbTON : TON;
    bDone : BOOL;
    tDuration : TIME;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fbTON(IN := TRUE, PT := T#5s, Q => bDone, ET => tDuration);
FC_OutputArgs := bDone;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_function_mixed_args(self, twincat_parser):
        """Test fb(a, IN := b, Q => c) mixed argument calls."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MixedArgs" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MixedArgs : INT
VAR
    fbCustom : FB_Custom;
    nInput : INT;
    nOutput : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nInput := 10;
fbCustom(nInput, bEnable := TRUE, nResult => nOutput);
FC_MixedArgs := nOutput;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_method_call(self, twincat_parser):
        """Test fbTimer.Reset() method calls."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MethodCall" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MethodCall : INT
VAR
    fbTimer : FB_Timer;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[fbTimer.Start();
fbTimer.Reset();
FC_MethodCall := 1;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_chained_method_call(self, twincat_parser):
        """Test stData.stNested.GetResult() chained method calls."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ChainedMethodCall" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ChainedMethodCall : INT
VAR
    stData : ST_ComplexData;
    nResult : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nResult := stData.stNested.GetResult();
FC_ChainedMethodCall := nResult;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationSpecificFeatures
# =============================================================================


class TestImplementationSpecificFeatures:
    """Test features specific to implementation sections."""

    def test_expression_statement(self, twincat_parser):
        """Test stData.nCounter; standalone access expression statement."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ExpressionStatement" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ExpressionStatement : INT
VAR
    stData : ST_Data;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[stData.nCounter;
FC_ExpressionStatement := 1;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_empty_statement(self, twincat_parser):
        """Test ; empty statement is valid."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_EmptyStatement" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_EmptyStatement : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 1;
;
;
FC_EmptyStatement := n;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_exit_in_loop(self, twincat_parser):
        """Test EXIT; inside FOR/WHILE loops."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ExitInLoop" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ExitInLoop : INT
VAR
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[FOR i := 0 TO 100 DO
    IF i > 50 THEN
        EXIT;
    END_IF;
END_FOR;
FC_ExitInLoop := i;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_continue_in_loop(self, twincat_parser):
        """Test CONTINUE; inside FOR loop."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ContinueInLoop" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ContinueInLoop : INT
VAR
    i : INT;
    nSum : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nSum := 0;
FOR i := 0 TO 10 DO
    IF i MOD 2 = 0 THEN
        CONTINUE;
    END_IF;
    nSum := nSum + i;
END_FOR;
FC_ContinueInLoop := nSum;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_return_statement(self, twincat_parser):
        """Test RETURN; and RETURN(value); statements."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ReturnStatement" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ReturnStatement : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 42;
IF n > 0 THEN
    RETURN;
END_IF;
FC_ReturnStatement := n;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_array_element_assignment(self, twincat_parser):
        """Test arr[i] := value; array element assignment."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ArrayAssignment" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ArrayAssignment : INT
VAR
    arr : ARRAY[0..9] OF INT;
    i : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[FOR i := 0 TO 9 DO
    arr[i] := i * 10;
END_FOR;
FC_ArrayAssignment := arr[5];
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation

    def test_member_assignment(self, twincat_parser):
        """Test stData.nValue := 42; member assignment."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_MemberAssignment" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_MemberAssignment : INT
VAR
    stData : ST_Data;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[stData.nValue := 42;
stData.sName := 'Test';
FC_MemberAssignment := stData.nValue;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        func_chunks = find_by_type(chunks, ChunkType.FUNCTION)
        assert len(func_chunks) == 2  # declaration + implementation


# =============================================================================
# TestImplementationWithComplexExpressions
# =============================================================================


class TestImplementationWithComplexExpressions:
    """Verify block extraction still works with complex expressions inside."""

    def test_if_with_type_conversion_condition(self, twincat_parser):
        """Test block extracted when condition has type conversion."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_IfTypeConversion" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_IfTypeConversion : INT
VAR
    nInt : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nInt := 42;
IF INT_TO_REAL(nInt) > 10.0 THEN
    FC_IfTypeConversion := nInt;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        if_blocks = find_by_metadata(block_chunks, "kind", "if_block")
        assert len(if_blocks) == 1
        assert if_blocks[0].metadata["pou_name"] == "FC_IfTypeConversion"

    def test_for_with_array_access_body(self, twincat_parser):
        """Test block extracted with array ops in body."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_ForArrayAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_ForArrayAccess : INT
VAR
    arr : ARRAY[0..9] OF INT;
    i : INT;
    nSum : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[nSum := 0;
FOR i := 0 TO 9 DO
    arr[i] := i * 10;
    nSum := nSum + arr[i];
END_FOR;
FC_ForArrayAccess := nSum;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        for_loops = find_by_metadata(block_chunks, "kind", "for_loop")
        assert len(for_loops) == 1
        assert for_loops[0].metadata["pou_type"] == "FUNCTION"

    def test_case_with_member_access_expression(self, twincat_parser):
        """Test block extracted with struct access in CASE expression."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_CaseMemberAccess" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_CaseMemberAccess : INT
VAR
    stData : ST_Data;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[stData.nMode := 2;
CASE stData.nMode OF
    1: FC_CaseMemberAccess := 10;
    2: FC_CaseMemberAccess := 20;
ELSE
    FC_CaseMemberAccess := 0;
END_CASE;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        case_blocks = find_by_metadata(block_chunks, "kind", "case_block")
        assert len(case_blocks) == 1

    def test_while_with_complex_condition(self, twincat_parser):
        """Test block extracted with compound condition in WHILE."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_WhileComplex" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_WhileComplex : INT
VAR
    i : INT;
    bContinue : BOOL;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[i := 0;
bContinue := TRUE;
WHILE (i < 100) AND bContinue DO
    i := i + 1;
    IF i > 50 THEN
        bContinue := FALSE;
    END_IF;
END_WHILE;
FC_WhileComplex := i;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        while_loops = find_by_metadata(block_chunks, "kind", "while_loop")
        assert len(while_loops) == 1
        # Also verify the nested IF is extracted
        if_blocks = find_by_metadata(block_chunks, "kind", "if_block")
        assert len(if_blocks) == 1


# =============================================================================
# TestImplementationBlockLineNumbers
# =============================================================================


class TestImplementationBlockLineNumbers:
    """Verify BLOCK chunks from Implementation have correct XML-absolute line numbers."""

    @pytest.fixture
    def function_if_fixture(self):
        """Load the FUNCTION IF test fixture."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "twincat" / "example_function_if.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION IF fixture not found")
        return fixture_path

    @pytest.fixture
    def function_loops_fixture(self):
        """Load the FUNCTION loops test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_loops.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION loops fixture not found")
        return fixture_path

    @pytest.fixture
    def function_action_fixture(self):
        """Load the FUNCTION with action test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_with_action.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("FUNCTION with action fixture not found")
        return fixture_path

    def test_if_block_start_line_is_xml_absolute(self, twincat_parser):
        """Test IF block start_line > CDATA start (not starting at 1)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 1;
IF n > 0 THEN
    FC_Test := n;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        # Implementation CDATA starts at line 10, IF is on line 11
        # Line numbers must be > 3 (XML declaration, TcPlcObject, POU elements)
        assert block_chunks[0].start_line > 10

    def test_if_block_end_line_spans_content(self, twincat_parser):
        """Test IF block end_line includes END_IF."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[n := 1;
IF n > 0 THEN
    FC_Test := n;
END_IF;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) == 1
        # end_line must be >= start_line and include END_IF
        assert block_chunks[0].end_line >= block_chunks[0].start_line
        # IF spans 3 lines (IF, body, END_IF), so end >= start + 2
        assert block_chunks[0].end_line >= block_chunks[0].start_line + 2

    def test_for_loop_line_numbers_match_xml_positions(
        self, twincat_parser, function_loops_fixture
    ):
        """Test specific FOR loop line numbers verified against known XML positions."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        for_loops = find_by_metadata(block_chunks, "kind", "for_loop")
        assert len(for_loops) >= 1
        # FOR loop in fixture starts at line 21 (inside CASE branch 1)
        # Implementation CDATA starts at line 16
        for for_loop in for_loops:
            assert for_loop.start_line > 16
            assert for_loop.end_line >= for_loop.start_line

    def test_nested_blocks_have_distinct_line_numbers(
        self, twincat_parser, function_if_fixture
    ):
        """Test outer IF and nested IF have different, correct line numbers."""
        chunks = extract_chunks_from_file(twincat_parser, function_if_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        if_blocks = find_by_metadata(block_chunks, "kind", "if_block")
        assert len(if_blocks) == 2
        # Sort by start_line to identify outer vs inner
        if_blocks_sorted = sorted(if_blocks, key=lambda x: x.start_line)
        outer_if = if_blocks_sorted[0]
        inner_if = if_blocks_sorted[1]
        # Nested IF must start after outer IF starts
        assert inner_if.start_line > outer_if.start_line
        # Nested IF must end before or at outer IF end
        assert inner_if.end_line <= outer_if.end_line

    def test_case_block_line_numbers(self, twincat_parser, function_loops_fixture):
        """Test CASE block start/end lines are XML-absolute."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        case_blocks = find_by_metadata(block_chunks, "kind", "case_block")
        assert len(case_blocks) == 1
        # CASE in fixture starts at line 19
        assert case_blocks[0].start_line >= 19
        assert case_blocks[0].end_line >= case_blocks[0].start_line

    def test_while_loop_line_numbers(self, twincat_parser, function_loops_fixture):
        """Test WHILE block line numbers are verified."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        while_loops = find_by_metadata(block_chunks, "kind", "while_loop")
        assert len(while_loops) >= 1
        for while_loop in while_loops:
            # Must be XML-absolute (> declaration CDATA end around line 14)
            assert while_loop.start_line > 14
            assert while_loop.end_line >= while_loop.start_line

    def test_repeat_loop_line_numbers(self, twincat_parser, function_loops_fixture):
        """Test REPEAT block line numbers are verified."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        repeat_loops = find_by_metadata(block_chunks, "kind", "repeat_loop")
        assert len(repeat_loops) >= 1
        for repeat_loop in repeat_loops:
            assert repeat_loop.start_line > 14
            assert repeat_loop.end_line >= repeat_loop.start_line

    def test_action_block_line_numbers_are_xml_absolute(
        self, twincat_parser, function_action_fixture
    ):
        """Test blocks in ACTION have XML-absolute lines (not action-relative)."""
        chunks = extract_chunks_from_file(twincat_parser, function_action_fixture)
        assert_no_parse_errors(twincat_parser)
        # Get action blocks only
        action_blocks = [
            c
            for c in chunks
            if c.concept == UniversalConcept.BLOCK and c.metadata.get("action_name")
        ]
        assert len(action_blocks) >= 1
        for block in action_blocks:
            # Action Implementation CDATA starts around line 30
            # Blocks must have line numbers > this
            assert block.start_line > 25
            assert block.end_line >= block.start_line

    def test_multiple_blocks_sequential_line_numbers(
        self, twincat_parser, function_loops_fixture
    ):
        """Test multiple blocks have increasing, non-overlapping line numbers."""
        chunks = extract_chunks_from_file(twincat_parser, function_loops_fixture)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        # Sort by start_line
        sorted_blocks = sorted(block_chunks, key=lambda x: x.start_line)
        # All blocks should have valid line ranges
        for block in sorted_blocks:
            assert block.start_line > 0
            assert block.end_line >= block.start_line

    def test_block_line_numbers_not_cdata_relative(self, twincat_parser):
        """Test all block line numbers > 3 (XML structure takes lines 1-3)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FC_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION FC_Test : INT
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF n > 0 THEN
    n := 1;
END_IF;
FOR n := 0 TO 5 DO
    n := n + 1;
END_FOR;
]]></ST>
    </Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        block_chunks = find_by_type(chunks, ChunkType.BLOCK)
        assert len(block_chunks) >= 2
        for block in block_chunks:
            # Line numbers must be > 3 because:
            # - Line 1: XML declaration
            # - Line 2: TcPlcObject
            # - Line 3: POU element
            # - Line 4+: Declaration CDATA
            # - Line N+: Implementation CDATA (where blocks are)
            assert block.start_line > 3, (
                f"Block {block.metadata.get('kind')} has start_line={block.start_line}, "
                "which suggests CDATA-relative numbering"
            )


# =============================================================================
# TestMethodExtraction
# =============================================================================


class TestMethodExtraction:
    """Test METHOD extraction from FUNCTION_BLOCK."""

    @pytest.fixture
    def method_fixture(self):
        """Load the methods test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_block_with_methods.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("Methods fixture not found")
        return fixture_path

    def test_method_chunk_created(self, twincat_parser, method_fixture):
        """Test METHOD creates separate declaration/implementation chunks."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)
        method_chunks = find_by_type(chunks, ChunkType.METHOD)
        # 3 methods x 2 chunks each = 6 total
        assert len(method_chunks) == 6
        # Check for declaration/implementation suffixes
        decl_names = {
            c.name for c in method_chunks if c.metadata.get("section") == "declaration"
        }
        impl_names = {
            c.name
            for c in method_chunks
            if c.metadata.get("section") == "implementation"
        }
        assert "FB_WithMethods.Initialize.declaration" in decl_names
        assert "FB_WithMethods.Initialize.implementation" in impl_names

    def test_method_metadata(self, twincat_parser, method_fixture):
        """Test METHOD metadata includes kind='method', pou_name, method_id, section."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)
        method_chunks = find_by_type(chunks, ChunkType.METHOD)
        init_decl = [
            c
            for c in method_chunks
            if c.name == "FB_WithMethods.Initialize.declaration"
        ][0]

        assert init_decl.metadata["kind"] == "method"
        assert init_decl.metadata["pou_type"] == "FUNCTION_BLOCK"
        assert init_decl.metadata["pou_name"] == "FB_WithMethods"
        assert (
            init_decl.metadata["method_id"] == "{11111111-2222-3333-4444-555555555555}"
        )
        assert init_decl.metadata["section"] == "declaration"

    def test_method_variables_extracted(self, twincat_parser, method_fixture):
        """Test METHOD variable declarations are extracted as FIELD chunks."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)

        # Variables from Initialize method: bReset, nInitValue, bLocalFlag
        init_vars = find_by_metadata(chunks, "method_name", "Initialize")
        assert len(init_vars) >= 3

        # Check that bReset is a VAR_INPUT with proper metadata
        breset = [c for c in init_vars if c.name == "FB_WithMethods.Initialize.bReset"]
        assert len(breset) == 1
        assert breset[0].concept == UniversalConcept.DEFINITION
        assert breset[0].metadata["kind"] == "field"
        assert breset[0].metadata["var_class"] == "input"
        assert breset[0].metadata["data_type"] == "BOOL"
        assert breset[0].metadata["method_name"] == "Initialize"

    def test_method_blocks_extracted(self, twincat_parser, method_fixture):
        """Test METHOD control flow blocks are extracted as BLOCK chunks."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)

        # Initialize method has an IF block
        init_blocks = [
            c
            for c in chunks
            if c.concept == UniversalConcept.BLOCK
            and c.metadata.get("method_name") == "Initialize"
        ]
        assert len(init_blocks) >= 1
        assert init_blocks[0].metadata["kind"] == "if_block"
        # Verify FQN symbol pattern: POUName.MethodName.{kind}_{line}
        assert init_blocks[0].name.startswith("FB_WithMethods.Initialize.if_block_")

    def test_method_code_separate_declaration_and_implementation(
        self, twincat_parser, method_fixture
    ):
        """Test METHOD has separate declaration and implementation chunks."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)
        method_chunks = find_by_type(chunks, ChunkType.METHOD)
        init_decl = [
            c
            for c in method_chunks
            if c.name == "FB_WithMethods.Initialize.declaration"
        ][0]
        init_impl = [
            c
            for c in method_chunks
            if c.name == "FB_WithMethods.Initialize.implementation"
        ][0]

        # Declaration should include METHOD header and VAR blocks
        assert "METHOD Initialize" in init_decl.content
        assert "VAR_INPUT" in init_decl.content
        assert "bReset" in init_decl.content

        # Implementation should include the IF block
        assert "IF bReset THEN" in init_impl.content

    def test_multiple_methods_have_unique_ids(self, twincat_parser, method_fixture):
        """Test each METHOD pair shares the same method_id."""
        chunks = extract_chunks_from_file(twincat_parser, method_fixture)
        assert_no_parse_errors(twincat_parser)
        method_chunks = find_by_type(chunks, ChunkType.METHOD)
        # 3 unique method_ids, each appearing twice (decl + impl)
        method_ids = [c.metadata["method_id"] for c in method_chunks]
        assert len(set(method_ids)) == 3

    def test_method_from_xml_string(self, twincat_parser):
        """Test METHOD extraction from inline XML."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nState : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Method Name="DoWork" Id="{method-uuid}">
      <Declaration><![CDATA[METHOD DoWork : BOOL
VAR_INPUT
    nValue : INT;
END_VAR
]]></Declaration>
      <Implementation>
        <ST><![CDATA[DoWork := nValue > 0;]]></ST>
      </Implementation>
    </Method>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        method_chunks = find_by_type(chunks, ChunkType.METHOD)
        # 2 chunks: declaration + implementation
        assert len(method_chunks) == 2
        chunk_names = {c.name for c in method_chunks}
        assert chunk_names == {
            "FB_Test.DoWork.declaration",
            "FB_Test.DoWork.implementation",
        }
        # Both share same method_id
        for c in method_chunks:
            assert c.metadata["method_id"] == "{method-uuid}"


# =============================================================================
# TestPropertyExtraction
# =============================================================================


class TestPropertyExtraction:
    """Test PROPERTY extraction from FUNCTION_BLOCK."""

    @pytest.fixture
    def property_fixture(self):
        """Load the properties test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_function_block_with_properties.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("Properties fixture not found")
        return fixture_path

    def test_property_chunk_created(self, twincat_parser, property_fixture):
        """Test PROPERTY creates separate declaration/get/set chunks."""
        chunks = extract_chunks_from_file(twincat_parser, property_fixture)
        assert_no_parse_errors(twincat_parser)
        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        # Value: decl + get + set = 3, ReadOnlyStatus: decl + get = 2, Name: decl + get + set = 3 = 8 total
        assert len(property_chunks) >= 7
        # Check for section suffixes
        sections = {c.metadata.get("section") for c in property_chunks}
        assert "declaration" in sections
        assert "get" in sections

    def test_property_metadata(self, twincat_parser, property_fixture):
        """Test PROPERTY metadata includes kind='property', pou_name, property_id, section."""
        chunks = extract_chunks_from_file(twincat_parser, property_fixture)
        assert_no_parse_errors(twincat_parser)
        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        value_decl = [
            c
            for c in property_chunks
            if c.name == "FB_WithProperties.Value.declaration"
        ][0]

        assert value_decl.metadata["kind"] == "property"
        assert value_decl.metadata["pou_type"] == "FUNCTION_BLOCK"
        assert value_decl.metadata["pou_name"] == "FB_WithProperties"
        assert (
            value_decl.metadata["property_id"]
            == "{44444444-5555-6666-7777-888888888888}"
        )
        assert value_decl.metadata["section"] == "declaration"

    def test_property_separate_accessor_chunks(self, twincat_parser, property_fixture):
        """Test PROPERTY has separate declaration, get, and set chunks."""
        chunks = extract_chunks_from_file(twincat_parser, property_fixture)
        assert_no_parse_errors(twincat_parser)
        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        value_get = [
            c for c in property_chunks if c.name == "FB_WithProperties.Value.get"
        ]
        assert len(value_get) == 1
        assert "nInternal" in value_get[0].content

    def test_property_from_xml_string(self, twincat_parser):
        """Test PROPERTY extraction from inline XML."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nInternal : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Property Name="MyProp" Id="{prop-uuid}">
      <Declaration><![CDATA[PROPERTY MyProp : INT
]]></Declaration>
      <Get Name="Get" Id="{get-uuid}">
        <Declaration><![CDATA[VAR
END_VAR
]]></Declaration>
        <Implementation>
          <ST><![CDATA[MyProp := nInternal;]]></ST>
        </Implementation>
      </Get>
      <Set Name="Set" Id="{set-uuid}">
        <Declaration><![CDATA[VAR
END_VAR
]]></Declaration>
        <Implementation>
          <ST><![CDATA[nInternal := MyProp;]]></ST>
        </Implementation>
      </Set>
    </Property>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        # 3 chunks: declaration + get + set
        assert len(property_chunks) == 3
        chunk_names = {c.name for c in property_chunks}
        assert chunk_names == {
            "FB_Test.MyProp.declaration",
            "FB_Test.MyProp.get",
            "FB_Test.MyProp.set",
        }
        # All share same property_id
        for c in property_chunks:
            assert c.metadata["property_id"] == "{prop-uuid}"

    def test_readonly_property(self, twincat_parser):
        """Test read-only PROPERTY (GET only, no SET)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nInternal : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Property Name="ReadOnly" Id="{prop-uuid}">
      <Declaration><![CDATA[PROPERTY ReadOnly : INT
]]></Declaration>
      <Get Name="Get" Id="{get-uuid}">
        <Declaration><![CDATA[VAR
END_VAR
]]></Declaration>
        <Implementation>
          <ST><![CDATA[ReadOnly := nInternal;]]></ST>
        </Implementation>
      </Get>
    </Property>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        # 2 chunks: declaration + get (no set)
        assert len(property_chunks) == 2
        chunk_names = {c.name for c in property_chunks}
        assert chunk_names == {"FB_Test.ReadOnly.declaration", "FB_Test.ReadOnly.get"}

    def test_writeonly_property(self, twincat_parser):
        """Test write-only PROPERTY (SET only, no GET)."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    nInternal : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
    <Property Name="WriteOnly" Id="{prop-uuid}">
      <Declaration><![CDATA[PROPERTY WriteOnly : INT
]]></Declaration>
      <Set Name="Set" Id="{set-uuid}">
        <Declaration><![CDATA[VAR
END_VAR
]]></Declaration>
        <Implementation>
          <ST><![CDATA[nInternal := WriteOnly;]]></ST>
        </Implementation>
      </Set>
    </Property>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        property_chunks = find_by_type(chunks, ChunkType.PROPERTY)
        # 2 chunks: declaration + set (no get)
        assert len(property_chunks) == 2
        chunk_names = {c.name for c in property_chunks}
        assert chunk_names == {"FB_Test.WriteOnly.declaration", "FB_Test.WriteOnly.set"}


# =============================================================================
# TestChunkNameInSourceCode
# =============================================================================


class TestChunkNameInSourceCode:
    """Test that chunk names appear in their source code at start_line."""

    def test_comprehensive_chunk_names_in_start_line(
        self, twincat_parser, comprehensive_fixture
    ):
        """Verify each chunk's name component appears in code at its start_line.

        For every chunk, extract the meaningful segment of the symbol,
        and verify it exists in the line of code at start_line.

        Excludes:
        - BLOCK chunks: have synthetic names (if_block_XXX, for_loop_XXX)
        - POU chunks: start_line points to CDATA section start
        - Chunks with section suffixes (declaration, implementation, get, set)
        """
        # Parse the comprehensive fixture
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)

        # Read the source file to get individual lines
        content = comprehensive_fixture.read_text()
        lines = content.splitlines()

        # Chunk kinds to skip (synthetic names or special line handling)
        # Using metadata["kind"] for UniversalChunk
        skip_kinds = {
            # BLOCK chunks have synthetic names like if_block_295, for_loop_XYZ
            "if_block",
            "case_block",
            "for_loop",
            "while_loop",
            "repeat_loop",
            # COMMENT chunks have synthetic names like comment_line_4
            "comment",
            # POU chunks: start_line = CDATA start, not declaration line
            "function_block",
            "program",
            "function",
        }
        # Section suffixes that won't appear in source
        section_suffixes = {"declaration", "implementation", "get", "set"}

        for chunk in chunks:
            # Skip chunks with synthetic names or special line handling
            kind = chunk.metadata.get("kind", "")
            if kind in skip_kinds:
                continue
            # Also skip BLOCK concept entirely (covers control flow blocks)
            if chunk.concept == UniversalConcept.BLOCK:
                continue
            # Also skip COMMENT concept
            if chunk.concept == UniversalConcept.COMMENT:
                continue
            # Also skip IMPORT concept (FQN format is POUName:import_type:reference)
            if chunk.concept == UniversalConcept.IMPORT:
                continue

            # Extract the name (last segment after the final '.')
            name = chunk.name.rpartition(".")[2]
            # Skip if name is a section suffix (won't appear in source)
            if name in section_suffixes:
                continue

            # Get lines at start_line and start_line+1 (1-based indexing)
            line_index = chunk.start_line - 1
            assert 0 <= line_index < len(lines), (
                f"Chunk {chunk.name} has start_line {chunk.start_line} "
                f"which is out of range (file has {len(lines)} lines)"
            )

            line1 = lines[line_index]
            line2 = lines[line_index + 1] if line_index + 1 < len(lines) else ""

            # For variables (kind='field'), check first two lines to cover pragma attributes
            if kind == "field":
                assert name in line1 or name in line2, (
                    f"Variable name '{name}' (from name '{chunk.name}') "
                    f"not found in lines {chunk.start_line}-{chunk.start_line + 1}:\n"
                    f"  Line {chunk.start_line}: '{line1}'\n"
                    f"  Line {chunk.start_line + 1}: '{line2}'"
                )
            else:
                # For non-variable chunks, keep single-line check
                assert name in line1, (
                    f"Chunk name '{name}' (from name '{chunk.name}') "
                    f"not found in line {chunk.start_line}: '{line1}'"
                )


# =============================================================================
# TestCommentExtraction
# =============================================================================


class TestCommentExtraction:
    """Test comment chunk extraction from Structured Text code."""

    @pytest.fixture
    def comment_fixture(self):
        """Load the comment test fixture."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "twincat"
            / "example_with_comments.TcPOU"
        )
        if not fixture_path.exists():
            pytest.skip("Comment fixture not found")
        return fixture_path

    def test_comment_chunks_extracted(self, twincat_parser, comment_fixture):
        """Test that comment chunks are extracted from the fixture."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        assert len(comment_chunks) > 0, "Expected at least one comment chunk"

    def test_block_comment_type(self, twincat_parser, comment_fixture):
        """Test that block comments have comment_type='block' in metadata."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        block_comments = find_by_metadata(chunks, "comment_type", "block")
        assert len(block_comments) > 0, "Expected at least one block comment"
        for chunk in block_comments:
            assert "(*" in chunk.content and "*)" in chunk.content

    def test_line_comment_type(self, twincat_parser, comment_fixture):
        """Test that line comments have comment_type='line' in metadata."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        line_comments = find_by_metadata(chunks, "comment_type", "line")
        assert len(line_comments) > 0, "Expected at least one line comment"
        for chunk in line_comments:
            assert chunk.content.startswith("//")

    def test_comment_fqn_format(self, twincat_parser, comment_fixture):
        """Test that comment FQNs follow pattern: POUName.comment_line_N."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        for chunk in comment_chunks:
            assert ".comment_line_" in chunk.name
            assert chunk.name.startswith("FB_CommentExample.")

    def test_comment_has_cleaned_text(self, twincat_parser, comment_fixture):
        """Test that comment metadata includes cleaned_text without markers."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        for chunk in comment_chunks:
            assert "cleaned_text" in chunk.metadata
            cleaned = chunk.metadata["cleaned_text"]
            # Cleaned text should not have markers
            assert not cleaned.startswith("(*")
            assert not cleaned.startswith("//")

    def test_comment_metadata_fields(self, twincat_parser, comment_fixture):
        """Test that comment chunks have required metadata fields."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        for chunk in comment_chunks:
            assert chunk.metadata.get("kind") == "comment"
            assert "comment_type" in chunk.metadata
            assert "pou_name" in chunk.metadata
            assert "pou_type" in chunk.metadata
            assert "cleaned_text" in chunk.metadata

    def test_multiline_block_comment(self, twincat_parser, comment_fixture):
        """Test that multi-line block comments have correct end_line."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        block_comments = find_by_metadata(chunks, "comment_type", "block")
        # At least one multiline block comment should exist
        multiline = [c for c in block_comments if c.end_line > c.start_line]
        assert len(multiline) > 0, "Expected at least one multi-line block comment"

    def test_comment_language_node_type(self, twincat_parser, comment_fixture):
        """Test that all comment chunks have language_node_type indicating Lark origin."""
        chunks = extract_chunks_from_file(twincat_parser, comment_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        for chunk in comment_chunks:
            # UniversalChunk is language-agnostic but tracks origin
            assert chunk.language_node_type == "lark_comment"

    def test_comprehensive_fixture_has_comments(
        self, twincat_parser, comprehensive_fixture
    ):
        """Test that the comprehensive fixture also extracts comments."""
        chunks = extract_chunks_from_file(twincat_parser, comprehensive_fixture)
        assert_no_parse_errors(twincat_parser)
        comment_chunks = find_by_type(chunks, ChunkType.COMMENT)
        # Comprehensive fixture has comments
        assert len(comment_chunks) > 0, "Expected comments in comprehensive fixture"


# =============================================================================
# TestImportExtraction
# =============================================================================


def find_imports_by_type(
    chunks: list[UniversalChunk], import_type: str
) -> list[UniversalChunk]:
    """Filter chunks by concept=IMPORT and import_type metadata."""
    return [
        c
        for c in chunks
        if c.concept == UniversalConcept.IMPORT
        and c.metadata.get("import_type") == import_type
    ]


class TestImportExtraction:
    """Test IMPORT concept extraction (VAR_EXTERNAL, EXTENDS, IMPLEMENTS, type references)."""

    # --- VAR_EXTERNAL Tests ---

    def test_var_external_creates_import_chunk(self, twincat_parser):
        """Test VAR_EXTERNAL creates IMPORT chunk with import_type='var_external'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_EXTERNAL
    nGlobalCounter : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "var_external")
        assert len(import_chunks) == 1
        assert import_chunks[0].concept == UniversalConcept.IMPORT
        assert import_chunks[0].metadata["var_name"] == "nGlobalCounter"
        assert import_chunks[0].metadata["data_type"] == "DINT"

    def test_var_external_import_metadata(self, twincat_parser):
        """Test VAR_EXTERNAL import has all required metadata fields."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_EXTERNAL
    extValue : REAL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "var_external")
        assert len(import_chunks) == 1
        metadata = import_chunks[0].metadata
        assert metadata["kind"] == "import"
        assert metadata["import_type"] == "var_external"
        assert metadata["var_class"] == "external"
        assert metadata["pou_name"] == "FB_Test"
        assert metadata["pou_type"] == "FUNCTION_BLOCK"

    def test_multiple_var_external(self, twincat_parser):
        """Test multiple VAR_EXTERNAL declarations create multiple import chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_EXTERNAL
    nCounter : DINT;
    fTemperature : REAL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "var_external")
        assert len(import_chunks) == 2
        names = {c.metadata["var_name"] for c in import_chunks}
        assert names == {"nCounter", "fTemperature"}

    # --- EXTENDS Tests ---

    def test_extends_creates_import_chunk(self, twincat_parser):
        """Test EXTENDS clause creates IMPORT chunk with import_type='extends'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Child" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Child EXTENDS FB_Parent
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "extends")
        assert len(import_chunks) == 1
        assert import_chunks[0].concept == UniversalConcept.IMPORT
        assert import_chunks[0].metadata["base_type"] == "FB_Parent"
        assert import_chunks[0].metadata["target_type"] == "FB_Child"

    def test_extends_import_fqn(self, twincat_parser):
        """Test EXTENDS import FQN follows pattern: POUName:extends:BaseType."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Motor" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Motor EXTENDS FB_Device
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "extends")
        assert len(import_chunks) == 1
        assert import_chunks[0].name == "FB_Motor:extends:FB_Device"

    # --- IMPLEMENTS Tests ---

    def test_implements_creates_import_chunk(self, twincat_parser):
        """Test IMPLEMENTS clause creates IMPORT chunk with import_type='implements'."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Motor" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Motor IMPLEMENTS I_Motor
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "implements")
        assert len(import_chunks) == 1
        assert import_chunks[0].concept == UniversalConcept.IMPORT
        assert import_chunks[0].metadata["interface_name"] == "I_Motor"
        assert import_chunks[0].metadata["implementing_type"] == "FB_Motor"

    def test_multiple_implements(self, twincat_parser):
        """Test multiple IMPLEMENTS interfaces create multiple import chunks."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Motor" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Motor IMPLEMENTS I_Motor, I_Device, I_Runnable
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "implements")
        assert len(import_chunks) == 3
        interface_names = {c.metadata["interface_name"] for c in import_chunks}
        assert interface_names == {"I_Motor", "I_Device", "I_Runnable"}

    def test_extends_and_implements_together(self, twincat_parser):
        """Test EXTENDS and IMPLEMENTS can be used together."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Child" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Child EXTENDS FB_Parent IMPLEMENTS I_Child, I_Other
VAR
    n : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        extends_chunks = find_imports_by_type(chunks, "extends")
        assert len(extends_chunks) == 1
        assert extends_chunks[0].metadata["base_type"] == "FB_Parent"

        implements_chunks = find_imports_by_type(chunks, "implements")
        assert len(implements_chunks) == 2
        interface_names = {c.metadata["interface_name"] for c in implements_chunks}
        assert interface_names == {"I_Child", "I_Other"}

    # --- Type Reference Tests ---

    def test_user_type_reference_creates_import(self, twincat_parser):
        """Test user-defined type reference creates IMPORT chunk."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    fbMotor : FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        assert len(import_chunks) == 1
        assert import_chunks[0].metadata["referenced_type"] == "FB_Motor"
        assert import_chunks[0].metadata["var_name"] == "fbMotor"

    def test_primitive_types_not_imported(self, twincat_parser):
        """Test primitive types (BOOL, INT, REAL, etc.) don't create type_reference imports."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    bFlag : BOOL;
    nCount : INT;
    fValue : REAL;
    tDelay : TIME;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        # No imports for primitive types
        assert len(import_chunks) == 0

    def test_array_of_user_type_creates_import(self, twincat_parser):
        """Test ARRAY OF user-defined type creates type_reference import."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    aMotors : ARRAY[0..9] OF FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        assert len(import_chunks) == 1
        assert import_chunks[0].metadata["referenced_type"] == "FB_Motor"

    def test_pointer_to_user_type_creates_import(self, twincat_parser):
        """Test POINTER TO user-defined type creates type_reference import."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    pMotor : POINTER TO FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        assert len(import_chunks) == 1
        assert import_chunks[0].metadata["referenced_type"] == "FB_Motor"

    def test_reference_to_user_type_creates_import(self, twincat_parser):
        """Test REFERENCE TO user-defined type creates type_reference import."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    refMotor : REFERENCE TO FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        assert len(import_chunks) == 1
        assert import_chunks[0].metadata["referenced_type"] == "FB_Motor"

    def test_type_reference_deduplication(self, twincat_parser):
        """Test same user type referenced multiple times creates only one import."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    fbMotor1 : FB_Motor;
    fbMotor2 : FB_Motor;
    fbMotor3 : FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        # Only one import despite three usages
        assert len(import_chunks) == 1
        assert import_chunks[0].metadata["referenced_type"] == "FB_Motor"

    def test_type_reference_fqn_format(self, twincat_parser):
        """Test type_reference import FQN follows pattern: POUName:type_ref:TypeName."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR
    fbDevice : FB_Device;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)
        import_chunks = find_imports_by_type(chunks, "type_reference")
        assert len(import_chunks) == 1
        assert import_chunks[0].name == "FB_Test:type_ref:FB_Device"

    # --- Combined Import Tests ---

    def test_all_import_types_together(self, twincat_parser):
        """Test extraction of all import types in a single POU."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Child" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Child EXTENDS FB_Parent IMPLEMENTS I_Motor
VAR_EXTERNAL
    nGlobalCounter : DINT;
END_VAR
VAR
    fbDriver : FB_MotorDriver;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        # Check all import types present
        var_external = find_imports_by_type(chunks, "var_external")
        extends = find_imports_by_type(chunks, "extends")
        implements = find_imports_by_type(chunks, "implements")
        type_refs = find_imports_by_type(chunks, "type_reference")

        assert len(var_external) == 1
        assert var_external[0].metadata["var_name"] == "nGlobalCounter"

        assert len(extends) == 1
        assert extends[0].metadata["base_type"] == "FB_Parent"

        assert len(implements) == 1
        assert implements[0].metadata["interface_name"] == "I_Motor"

        assert len(type_refs) == 1
        assert type_refs[0].metadata["referenced_type"] == "FB_MotorDriver"

    def test_import_chunks_have_language_node_type(self, twincat_parser):
        """Test all import chunks have appropriate language_node_type."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Child" Id="{1234}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Child EXTENDS FB_Parent IMPLEMENTS I_Motor
VAR_EXTERNAL
    nCounter : DINT;
END_VAR
VAR
    fbMotor : FB_Motor;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>"""
        chunks = twincat_parser.extract_universal_chunks(xml)
        assert_no_parse_errors(twincat_parser)

        import_chunks = [c for c in chunks if c.concept == UniversalConcept.IMPORT]
        assert len(import_chunks) == 4  # var_external, extends, implements, type_ref

        node_types = {c.language_node_type for c in import_chunks}
        assert "lark_var_external" in node_types
        assert "lark_extends" in node_types
        assert "lark_implements" in node_types
        assert "lark_type_reference" in node_types


# =============================================================================
# Import Resolution Tests
# =============================================================================


class TestTwinCATImportResolution:
    """Test TwinCAT import path resolution."""

    @pytest.fixture
    def mapping(self):
        from chunkhound.parsers.twincat.twincat_mapping import TwinCATMapping

        return TwinCATMapping()

    def test_resolve_direct_symbol(self, tmp_path, mapping):
        """Test resolving a direct symbol name to .TcPOU file."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "FB_Motor", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_resolve_var_declaration(self, tmp_path, mapping):
        """Test resolving type from variable declaration."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "motor : FB_Motor;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_resolve_extends_keyword(self, tmp_path, mapping):
        """Test resolving type from EXTENDS clause."""
        base_file = tmp_path / "FB_Base.TcPOU"
        base_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "EXTENDS FB_Base", tmp_path, tmp_path / "FB_Child.TcPOU"
        )
        assert resolved == [base_file]

    def test_resolve_implements_keyword(self, tmp_path, mapping):
        """Test resolving type from IMPLEMENTS clause."""
        interface_file = tmp_path / "I_Runnable.TcPOU"
        interface_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "IMPLEMENTS I_Runnable", tmp_path, tmp_path / "FB_Motor.TcPOU"
        )
        assert resolved == [interface_file]

    def test_resolve_pointer_to_type(self, tmp_path, mapping):
        """Test resolving type from POINTER TO declaration."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "pMotor : POINTER TO FB_Motor;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_resolve_reference_to_type(self, tmp_path, mapping):
        """Test resolving type from REFERENCE TO declaration."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "refMotor : REFERENCE TO FB_Motor;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_resolve_array_of_type(self, tmp_path, mapping):
        """Test resolving type from ARRAY OF declaration."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "motors : ARRAY[0..9] OF FB_Motor;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_resolve_nested_pointer_array(self, tmp_path, mapping):
        """Test resolving type from nested POINTER TO ARRAY OF declaration."""
        motor_file = tmp_path / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "pMotors : POINTER TO ARRAY[0..9] OF FB_Motor;",
            tmp_path,
            tmp_path / "Main.TcPOU",
        )
        assert resolved == [motor_file]

    def test_case_insensitive_matching(self, tmp_path, mapping):
        """Test case-insensitive file matching."""
        # Create file with different case than symbol
        motor_file = tmp_path / "fb_motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "FB_MOTOR", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]

    def test_primitive_type_returns_empty_list(self, tmp_path, mapping):
        """Test that IEC 61131-3 primitive types return empty list."""
        resolved = mapping.resolve_import_paths(
            "value : DINT;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == []

    def test_stdlib_type_returns_empty_list(self, tmp_path, mapping):
        """Test that standard library types (TON, etc.) return empty list."""
        resolved = mapping.resolve_import_paths(
            "timer : TON;", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == []

    def test_not_found_returns_empty_list(self, tmp_path, mapping):
        """Test that non-existent symbols return empty list."""
        resolved = mapping.resolve_import_paths(
            "FB_NonExistent", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == []

    def test_searches_subdirectories(self, tmp_path, mapping):
        """Test that file search includes subdirectories."""
        lib_dir = tmp_path / "lib" / "motors"
        lib_dir.mkdir(parents=True)
        motor_file = lib_dir / "FB_Motor.TcPOU"
        motor_file.write_text("<xml/>")

        resolved = mapping.resolve_import_paths(
            "FB_Motor", tmp_path, tmp_path / "Main.TcPOU"
        )
        assert resolved == [motor_file]
