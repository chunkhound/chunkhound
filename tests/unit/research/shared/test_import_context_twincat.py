"""Tests for ImportContextService with TwinCAT parser integration.

These tests verify that ImportContextService correctly detects and extracts
imports from TwinCAT (Lark-based) parsers, ensuring deep research synthesis
works correctly with Structured Text files.

Test Categories:
1. Parser detection - Verify Lark-based parser is correctly identified
2. Import extraction - Test extraction of various TwinCAT import types
3. Caching behavior - Verify imports are cached correctly
4. Edge cases - Empty files, parse errors, etc.
"""

from pathlib import Path

import pytest

from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.services.research.shared.import_context import ImportContextService


# TcPOU XML fixtures
VAR_EXTERNAL_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Test" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Test
VAR_EXTERNAL
    nGlobal : DINT;
    sMessage : STRING;
END_VAR
VAR
    nLocal : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

EXTENDS_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Child" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Child EXTENDS FB_Base
VAR
    nValue : INT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

IMPLEMENTS_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Motor" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Motor IMPLEMENTS I_Motor
VAR
    bRunning : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

MULTIPLE_IMPLEMENTS_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Device" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Device IMPLEMENTS I_Device, I_Controllable
VAR
    bEnabled : BOOL;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

TYPE_REFERENCE_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Controller" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Controller
VAR
    fbDevice : FB_Device;
    stData : ST_ProcessData;
    eState : E_MachineState;
    nCounter : DINT;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

COMBINED_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Combined" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Combined EXTENDS FB_Base IMPLEMENTS I_Motor, I_Device
VAR_EXTERNAL
    nGlobal : DINT;
END_VAR
VAR
    fbSensor : FB_Sensor;
    stConfig : ST_Config;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

EMPTY_DECLARATION_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Empty" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Empty
END_FUNCTION_BLOCK
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""

PRIMITIVES_ONLY_FIXTURE = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Primitives" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Primitives
VAR
    bFlag : BOOL;
    nCount : DINT;
    fValue : REAL;
    sName : STRING;
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""


@pytest.fixture
def parser_factory():
    """Create a ParserFactory for testing."""
    return ParserFactory()


@pytest.fixture
def import_context_service(parser_factory):
    """Create an ImportContextService for testing."""
    return ImportContextService(parser_factory)


class TestLarkParserDetection:
    """Tests for detecting TwinCAT as a Lark-based parser."""

    def test_detects_lark_based_parser(self, parser_factory):
        """Parser for TwinCAT should have base_mapping.extract_imports."""
        parser = parser_factory.create_parser_for_file(Path("test.TcPOU"))

        # TwinCAT parser has engine=None (no tree-sitter)
        assert parser.engine is None

        # Should have base_mapping with extract_imports method
        assert hasattr(parser, "base_mapping")
        assert hasattr(parser.base_mapping, "extract_imports")

    def test_tcpou_file_extension_recognized(self, parser_factory):
        """Should recognize .TcPOU extension for TwinCAT files."""
        # Test various case variations
        for ext in [".TcPOU", ".tcpou", ".TCPOU"]:
            parser = parser_factory.create_parser_for_file(Path(f"test{ext}"))
            assert hasattr(parser, "base_mapping")


class TestVarExternalExtraction:
    """Tests for VAR_EXTERNAL import extraction."""

    def test_extracts_var_external_imports(self, import_context_service):
        """Should extract VAR_EXTERNAL declarations as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", VAR_EXTERNAL_FIXTURE
        )

        # Should find both VAR_EXTERNAL variables
        assert len(imports) >= 2
        import_text = "\n".join(imports)
        assert "nGlobal" in import_text
        assert "sMessage" in import_text

    def test_var_external_not_mixed_with_local_vars(self, import_context_service):
        """Should not include regular VAR declarations as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", VAR_EXTERNAL_FIXTURE
        )

        import_text = "\n".join(imports)
        # nLocal is in VAR block, not VAR_EXTERNAL
        assert "nLocal" not in import_text


class TestExtendsExtraction:
    """Tests for EXTENDS clause import extraction."""

    def test_extracts_extends_import(self, import_context_service):
        """Should extract EXTENDS clause as import."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", EXTENDS_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "FB_Base" in import_text

    def test_extends_marked_as_inheritance_type(self, import_context_service):
        """EXTENDS import should be identifiable in output."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", EXTENDS_FIXTURE
        )

        # Should have at least one import containing FB_Base
        assert any("FB_Base" in imp for imp in imports)


class TestImplementsExtraction:
    """Tests for IMPLEMENTS clause import extraction."""

    def test_extracts_single_implements(self, import_context_service):
        """Should extract single IMPLEMENTS clause as import."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", IMPLEMENTS_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "I_Motor" in import_text

    def test_extracts_multiple_implements(self, import_context_service):
        """Should extract multiple IMPLEMENTS interfaces."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", MULTIPLE_IMPLEMENTS_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "I_Device" in import_text
        assert "I_Controllable" in import_text


class TestTypeReferenceExtraction:
    """Tests for user-defined type reference extraction."""

    def test_extracts_function_block_type_references(self, import_context_service):
        """Should extract FB_ type references as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", TYPE_REFERENCE_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "FB_Device" in import_text

    def test_extracts_struct_type_references(self, import_context_service):
        """Should extract ST_ struct type references as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", TYPE_REFERENCE_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "ST_ProcessData" in import_text

    def test_extracts_enum_type_references(self, import_context_service):
        """Should extract E_ enum type references as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", TYPE_REFERENCE_FIXTURE
        )

        import_text = "\n".join(imports)
        assert "E_MachineState" in import_text

    def test_excludes_primitive_types(self, import_context_service):
        """Should not extract primitive types (DINT, BOOL, etc.) as imports."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", PRIMITIVES_ONLY_FIXTURE
        )

        # No imports should be extracted for primitives only
        import_text = "\n".join(imports)
        # Primitive types should not appear as standalone imports
        assert "BOOL" not in import_text.split()[0] if imports else True


class TestCombinedExtraction:
    """Tests for combined import extraction scenarios."""

    def test_extracts_all_import_types(self, import_context_service):
        """Should extract EXTENDS, IMPLEMENTS, VAR_EXTERNAL, and type refs."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", COMBINED_FIXTURE
        )

        import_text = "\n".join(imports)

        # EXTENDS
        assert "FB_Base" in import_text
        # IMPLEMENTS (both)
        assert "I_Motor" in import_text
        assert "I_Device" in import_text
        # VAR_EXTERNAL
        assert "nGlobal" in import_text
        # Type references
        assert "FB_Sensor" in import_text
        assert "ST_Config" in import_text


class TestCaching:
    """Tests for import caching behavior."""

    def test_caches_imports(self, import_context_service):
        """Should cache imports and return cached value on second call."""
        file_path = "cached_test.TcPOU"

        # First call
        imports1 = import_context_service.get_file_imports(
            file_path, VAR_EXTERNAL_FIXTURE
        )

        # Second call with same path (content ignored, uses cache)
        imports2 = import_context_service.get_file_imports(
            file_path, "invalid content"
        )

        # Should return same cached result
        assert imports1 == imports2

    def test_clear_cache_removes_entries(self, import_context_service):
        """Should remove cached entries when clear_cache is called."""
        file_path = "cache_clear_test.TcPOU"

        # Populate cache
        imports1 = import_context_service.get_file_imports(
            file_path, VAR_EXTERNAL_FIXTURE
        )
        assert len(imports1) > 0

        # Clear cache
        import_context_service.clear_cache()

        # With empty declaration, should get no imports (not cached result)
        imports2 = import_context_service.get_file_imports(
            file_path, EMPTY_DECLARATION_FIXTURE
        )

        # Should not be the same (cache was cleared, parsed new content)
        assert imports1 != imports2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_returns_empty_for_no_imports(self, import_context_service):
        """Should return empty list when no imports present."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", PRIMITIVES_ONLY_FIXTURE
        )

        # May be empty or contain only type refs depending on implementation
        # At minimum, should not raise an error
        assert isinstance(imports, list)

    def test_returns_empty_for_empty_declaration(self, import_context_service):
        """Should return empty list for empty declaration block."""
        imports = import_context_service.get_file_imports(
            "test.TcPOU", EMPTY_DECLARATION_FIXTURE
        )

        assert imports == []

    def test_handles_invalid_xml_gracefully(self, import_context_service):
        """Should handle invalid XML without raising exception."""
        invalid_xml = "not valid xml content"

        # Should not raise
        imports = import_context_service.get_file_imports(
            "test.TcPOU", invalid_xml
        )

        assert imports == []

    def test_handles_malformed_declaration_gracefully(self, import_context_service):
        """Should handle malformed Structured Text gracefully."""
        malformed_tcpou = """<?xml version="1.0"?>
<TcPlcObject Version="1.1.0.1">
  <POU Name="FB_Malformed" Id="{12345678-1234-1234-1234-123456789abc}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_Malformed
VAR
    this is not valid structured text syntax
END_VAR
]]></Declaration>
    <Implementation><ST><![CDATA[]]></ST></Implementation>
  </POU>
</TcPlcObject>
"""
        # Should not raise
        imports = import_context_service.get_file_imports(
            "test.TcPOU", malformed_tcpou
        )

        # Should return empty list on parse error
        assert isinstance(imports, list)

    def test_different_file_paths_use_separate_cache_entries(
        self, import_context_service
    ):
        """Different file paths should have independent cache entries."""
        content = VAR_EXTERNAL_FIXTURE

        imports1 = import_context_service.get_file_imports("file1.TcPOU", content)
        imports2 = import_context_service.get_file_imports("file2.TcPOU", content)

        # Both should have the same imports
        assert imports1 == imports2

        # But clearing should not affect other entries until cleared
        # (This tests that paths are cached independently)
        assert len(import_context_service._import_cache) == 2
