"""Tests for LLM Provider interface utilities."""


from chunkhound.interfaces.llm_provider import _normalize_schema_for_structured_outputs


class TestNormalizeSchemaForStructuredOutputs:
    """Tests for the schema normalization helper function."""

    def test_adds_additional_properties_to_top_level(self):
        """Test that additionalProperties is added to top-level objects."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        result = _normalize_schema_for_structured_outputs(schema)

        assert result["additionalProperties"] is False

    def test_forces_additional_properties_false_with_warning(self, caplog):
        """Test that invalid additionalProperties is forced to false with warning.

        Anthropic's structured outputs API only supports additionalProperties: false.
        Any other value should be forced to false with a warning log.
        """
        import logging

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,  # Invalid - will be forced to false
        }

        with caplog.at_level(logging.WARNING):
            result = _normalize_schema_for_structured_outputs(schema)

        # Should force to false, not preserve
        assert result["additionalProperties"] is False

        # Should log a warning
        assert "additionalProperties=True not supported" in caplog.text
        assert "forcing to false" in caplog.text

    def test_adds_additional_properties_to_defs(self):
        """Test that additionalProperties is added to nested $defs."""
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                }
            },
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"},
            },
            "required": ["name", "address"],
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # Top level should have additionalProperties
        assert result["additionalProperties"] is False

        # Nested $defs should also have additionalProperties
        assert result["$defs"]["Address"]["additionalProperties"] is False

    def test_adds_additional_properties_to_array_items(self):
        """Test that additionalProperties is added to array items."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                }
            },
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # Top level
        assert result["additionalProperties"] is False

        # Array items
        assert result["properties"]["items"]["items"]["additionalProperties"] is False

    def test_adds_additional_properties_to_prefix_items(self):
        """Test that additionalProperties is added to prefixItems (Pydantic tuples)."""
        schema = {
            "type": "object",
            "properties": {
                "endpoints": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "object", "properties": {"x": {"type": "integer"}}},
                        {"type": "object", "properties": {"y": {"type": "integer"}}},
                    ],
                }
            },
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # Top level
        assert result["additionalProperties"] is False

        # prefixItems objects should have additionalProperties
        prefix_items = result["properties"]["endpoints"]["prefixItems"]
        assert prefix_items[0]["additionalProperties"] is False
        assert prefix_items[1]["additionalProperties"] is False

    def test_handles_any_of(self):
        """Test that additionalProperties is added to anyOf schemas."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                        },
                    ]
                }
            },
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # The object in anyOf should have additionalProperties
        any_of_schemas = result["properties"]["value"]["anyOf"]
        object_schema = next(s for s in any_of_schemas if s.get("type") == "object")
        assert object_schema["additionalProperties"] is False

    def test_handles_one_of(self):
        """Test that additionalProperties is added to oneOf schemas."""
        schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                }
            ]
        }

        result = _normalize_schema_for_structured_outputs(schema)

        assert result["oneOf"][0]["additionalProperties"] is False

    def test_handles_all_of(self):
        """Test that additionalProperties is added to allOf schemas."""
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"base": {"type": "string"}},
                },
                {
                    "type": "object",
                    "properties": {"extra": {"type": "string"}},
                },
            ]
        }

        result = _normalize_schema_for_structured_outputs(schema)

        assert result["allOf"][0]["additionalProperties"] is False
        assert result["allOf"][1]["additionalProperties"] is False

    def test_handles_non_object_types(self):
        """Test that non-object types are passed through unchanged."""
        schema = {"type": "string"}

        result = _normalize_schema_for_structured_outputs(schema)

        assert result == {"type": "string"}
        assert "additionalProperties" not in result

    def test_handles_non_dict_input(self):
        """Test that non-dict input is returned unchanged."""
        result = _normalize_schema_for_structured_outputs("not a dict")  # type: ignore

        assert result == "not a dict"

    def test_does_not_mutate_original(self):
        """Test that the original schema is not modified."""
        original = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        original_copy = original.copy()

        _normalize_schema_for_structured_outputs(original)

        # Original should be unchanged
        assert original == original_copy
        assert "additionalProperties" not in original

    def test_deeply_nested_objects(self):
        """Test normalization of deeply nested object structures."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                }
                            },
                        }
                    },
                }
            },
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # All levels should have additionalProperties
        assert result["additionalProperties"] is False
        assert result["properties"]["level1"]["additionalProperties"] is False
        assert (
            result["properties"]["level1"]["properties"]["level2"][
                "additionalProperties"
            ]
            is False
        )
        assert (
            result["properties"]["level1"]["properties"]["level2"]["properties"][
                "level3"
            ]["additionalProperties"]
            is False
        )

    def test_pydantic_model_schema_format(self):
        """Test with a typical Pydantic model schema format."""
        # This simulates what Pydantic generates for nested models
        schema = {
            "$defs": {
                "Item": {
                    "properties": {
                        "id": {"title": "Id", "type": "integer"},
                        "name": {"title": "Name", "type": "string"},
                    },
                    "required": ["id", "name"],
                    "title": "Item",
                    "type": "object",
                }
            },
            "properties": {
                "items": {
                    "items": {"$ref": "#/$defs/Item"},
                    "title": "Items",
                    "type": "array",
                }
            },
            "required": ["items"],
            "title": "Response",
            "type": "object",
        }

        result = _normalize_schema_for_structured_outputs(schema)

        # Top level should have additionalProperties
        assert result["additionalProperties"] is False

        # $defs/Item should have additionalProperties
        assert result["$defs"]["Item"]["additionalProperties"] is False
