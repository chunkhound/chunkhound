"""Direct tests for parse_and_validate_structured_json utility."""

import json

import pytest

from chunkhound.utils.json_extraction import (
    build_schema_system_instruction,
    extract_json_from_response,
    parse_and_validate_structured_json,
)


class TestExtractJsonFromResponse:
    """Cover markdown code block extraction edge cases."""

    def test_raw_json(self):
        content = '{"key": "value"}'
        assert extract_json_from_response(content) == '{"key": "value"}'

    def test_json_code_block(self):
        content = "```json\n{\"key\": \"value\"}\n```"
        assert extract_json_from_response(content) == '{"key": "value"}'

    def test_generic_code_block(self):
        content = "```\n{\"key\": \"value\"}\n```"
        assert extract_json_from_response(content) == '{"key": "value"}'

    def test_nested_code_blocks_takes_first(self):
        content = "```json\n{\"first\": 1}\n```\n```json\n{\"second\": 2}\n```"
        assert extract_json_from_response(content) == '{"first": 1}'

    def test_skips_non_json_code_block_before_valid_json(self):
        content = "```text\nnot json\n```\n```json\n{\"second\": 2}\n```"
        assert extract_json_from_response(content) == '{"second": 2}'

    def test_falls_back_to_raw_content_when_code_blocks_are_not_json(self):
        content = '```text\nnot json\n```\n{\"fallback\": true}'
        expected = '```text\nnot json\n```\n{"fallback": true}'
        assert extract_json_from_response(content) == expected

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="No JSON content found"):
            extract_json_from_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="No JSON content found"):
            extract_json_from_response("   \n\n  ")


class TestBuildSchemaSystemInstruction:
    """Verify the shared schema-instruction builder."""

    def test_includes_schema_text(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        instruction = build_schema_system_instruction(schema)
        assert "You must respond with valid JSON" in instruction
        assert '"name"' in instruction
        assert "no markdown formatting" in instruction

    def test_produces_valid_json_snippet(self):
        schema = {"type": "object"}
        instruction = build_schema_system_instruction(schema)
        # Extract JSON object from the instruction string
        import re

        m = re.search(r"\{.*\}", instruction, re.DOTALL)
        assert m is not None, f"No JSON object found in: {instruction!r}"
        parsed = json.loads(m.group())
        assert parsed == schema


class TestParseAndValidateStructuredJson:
    """Direct unit tests for the schema validator — independent of providers."""

    def test_valid_json_passes(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = parse_and_validate_structured_json('{"name": "Alice"}', schema)
        assert result == {"name": "Alice"}

    def test_markdown_code_block_extracted_then_validated(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        content = "```json\n{\"value\": 42}\n```"
        result = parse_and_validate_structured_json(content, schema)
        assert result == {"value": 42}

    def test_ignores_non_json_code_block_before_valid_json(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        content = "```text\nignore me\n```\n```json\n{\"value\": 42}\n```"
        result = parse_and_validate_structured_json(content, schema)
        assert result == {"value": 42}

    def test_rejects_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(ValueError, match="schema validation failed"):
            parse_and_validate_structured_json('{"age": 30}', schema)

    def test_rejects_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        with pytest.raises(ValueError, match="schema validation failed"):
            parse_and_validate_structured_json('{"count": "seven"}', schema)

    def test_rejects_additional_properties_when_forbidden(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="schema validation failed"):
            parse_and_validate_structured_json(
                '{"name": "Alice", "extra": true}', schema
            )

    def test_rejects_nested_schema_violation(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="schema validation failed"):
            parse_and_validate_structured_json(
                '{"items": [{"label": 7}]}', schema
            )

    def test_error_includes_json_path(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(ValueError) as exc:
            parse_and_validate_structured_json('{"name": 123}', schema)

        msg = str(exc.value)
        assert "schema validation failed" in msg
        assert "name" in msg

    def test_rejects_non_dict_top_level(self):
        schema = {"type": "object"}
        with pytest.raises(ValueError, match="Expected JSON object"):
            parse_and_validate_structured_json('[1, 2, 3]', schema)

    def test_rejects_invalid_format(self):
        schema = {
            "type": "object",
            "properties": {"email": {"type": "string", "format": "email"}},
            "required": ["email"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="schema validation failed"):
            parse_and_validate_structured_json('{"email": "not-an-email"}', schema)

    def test_rejects_invalid_json(self):
        schema = {"type": "object"}
        with pytest.raises(json.JSONDecodeError):
            parse_and_validate_structured_json('not json at all', schema)

    def test_nested_object_passes(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                }
            },
            "required": ["user"],
            "additionalProperties": False,
        }
        result = parse_and_validate_structured_json(
            '{"user": {"name": "Alice", "age": 30}}', schema
        )
        assert result["user"]["name"] == "Alice"
        assert result["user"]["age"] == 30

    def test_array_of_strings_passes(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tags"],
        }
        result = parse_and_validate_structured_json(
            '{"tags": ["python", "testing"]}', schema
        )
        assert result["tags"] == ["python", "testing"]
