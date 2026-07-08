"""Shared utilities for extracting JSON from LLM responses."""

import json
import re
from typing import Any

from jsonschema import FormatChecker
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for


def extract_json_from_response(content: str) -> str:
    """Extract JSON from response, handling markdown code blocks.

    Handles multiple patterns:
    - Raw JSON (no code blocks)
    - JSON in ```json code block
    - JSON in generic ``` code block
    - Nested code blocks (takes the first valid one)

    Args:
        content: Response content potentially containing JSON

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no valid JSON content can be extracted
    """
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(code_block_pattern, content, re.DOTALL)

    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            continue

    json_content = content.strip()
    if not json_content:
        raise ValueError("No JSON content found in response")

    return json_content


def _extract_and_parse_json(content: str) -> Any:
    """Extract and parse JSON from response, returning a Python object."""
    return json.loads(extract_json_from_response(content))


def build_schema_system_instruction(json_schema: dict[str, Any]) -> str:
    """Build a system-prompt instruction that injects a JSON schema.

    Used as a fallback when the LLM API does not support native
    ``response_format`` with ``type: "json_schema"``.

    Args:
        json_schema: JSON Schema dictionary describing the expected output

    Returns:
        Instruction string to append to the system prompt
    """
    schema_text = json.dumps(json_schema, indent=2)
    return (
        "You must respond with valid JSON that conforms to this "
        f"schema:\n{schema_text}\n\n"
        "Respond with JSON only. No additional text, "
        "no markdown formatting."
    )


def parse_and_validate_structured_json(
    content: str, json_schema: dict[str, Any]
) -> dict[str, Any]:
    """Parse and validate structured JSON output against a schema.

    Args:
        content: Raw response content potentially containing JSON
        json_schema: JSON Schema describing the expected output

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If the extracted content is not valid JSON
        ValueError: If the parsed JSON does not conform to the schema
    """
    parsed = _extract_and_parse_json(content)

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    validator_cls = validator_for(json_schema)
    validator = validator_cls(json_schema, format_checker=FormatChecker())
    try:
        validator.validate(parsed)
    except ValidationError as exc:
        error_path = ".".join(str(part) for part in exc.absolute_path)
        location = f" at {error_path}" if error_path else ""
        raise ValueError(
            f"Structured output schema validation failed{location}: {exc.message}"
        ) from exc

    return parsed
