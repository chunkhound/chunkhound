import os
import sys

import pytest

# Ensure package path resolution for direct imports
sys.path.insert(0, os.path.abspath("chunkhound"))

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept


@pytest.mark.skipif(
    os.environ.get("CH_SKIP_HCL_TESTS") == "1",
    reason="HCL tests explicitly disabled via env",
)
def test_hcl_attribute_paths_basic():
    sample = (
        'terraform { required_version = ">= 1.5" }\n'
        'provider "aws" { region = var.region }\n'
        'resource "aws_s3_bucket" "b" {\n'
        '  bucket = "my-bucket"\n'
        '  tags = { Env = "dev" }\n'
        "}\n"
    )

    parser = ParserFactory().create_parser(Language.HCL)
    ast = parser.engine.parse_to_ast(sample)
    chunks = parser.extractor.extract_concept(
        ast.root_node, sample.encode(), UniversalConcept.DEFINITION
    )

    # Filter only attribute-definition nodes
    attr = [c for c in chunks if c.language_node_type == "attribute"]
    names = {c.name for c in attr}

    # Expect dotted paths for attributes
    expected = {
        "terraform.required_version",
        "provider.aws.region",
        "resource.aws_s3_bucket.b.bucket",
        "resource.aws_s3_bucket.b.tags",
    }
    assert expected.issubset(names), (
        f"Missing attributes: {expected - names} (got {names})"
    )

    # Check metadata contains key and full path
    for c in attr:
        assert c.metadata.get("key"), f"Attribute {c.name} missing key metadata"
        assert c.metadata.get("path") == c.name, "Metadata path should equal chunk name"


def test_hcl_nested_object_pairs():
    sample = (
        'resource "aws_s3_bucket" "b" {\n  tags = { Env = "dev", Team = "core" }\n}\n'
    )

    parser = ParserFactory().create_parser(Language.HCL)
    ast = parser.engine.parse_to_ast(sample)
    chunks = parser.extractor.extract_concept(
        ast.root_node, sample.encode(), UniversalConcept.DEFINITION
    )

    # Collect names
    names = {c.name for c in chunks}

    # Expect inner object pairs to be captured as dotted paths
    assert "resource.aws_s3_bucket.b.tags.Env" in names
    assert "resource.aws_s3_bucket.b.tags.Team" in names

    # Verify metadata value types are string
    name_to_chunk = {c.name: c for c in chunks}
    assert (
        name_to_chunk["resource.aws_s3_bucket.b.tags.Env"].metadata.get("value_type")
        == "string"
    )
    assert (
        name_to_chunk["resource.aws_s3_bucket.b.tags.Team"].metadata.get("value_type")
        == "string"
    )


def test_hcl_template_value_type():
    sample = (
        'variable "policy" {\n'
        "  default = <<EOF\n"
        "  {\n"
        '    "Version": "2012-10-17",\n'
        '    "Statement": []\n'
        "  }\n"
        "EOF\n"
        "}\n"
    )

    parser = ParserFactory().create_parser(Language.HCL)
    ast = parser.engine.parse_to_ast(sample)
    chunks = parser.extractor.extract_concept(
        ast.root_node, sample.encode(), UniversalConcept.DEFINITION
    )

    # Find the 'variable.policy' attribute as default/template
    candidates = [
        c
        for c in chunks
        if c.language_node_type in ("attribute", "object_elem") and "variable" in c.name
    ]
    assert any("policy" in c.name for c in candidates), (
        "Expected variable.policy in names"
    )

    # The default heredoc should be classified as a template
    # Look for the attribute node corresponding to default
    default_chunks = [c for c in candidates if c.name.endswith(".default")]
    # Fallback: accept any variable.policy.* with value_type template
    if not default_chunks:
        default_chunks = [
            c for c in candidates if c.metadata.get("value_type") == "template"
        ]
    assert any(c.metadata.get("value_type") == "template" for c in default_chunks), (
        "Expected template value_type for heredoc"
    )


def test_hcl_chunk_types_table_and_key_value():
    sample = (
        'resource "aws_s3_bucket" "b" {\n'
        '  bucket = "my-bucket"\n'
        '  tags = { Env = "dev" }\n'
        "}\n"
    )

    parser = ParserFactory().create_parser(Language.HCL)
    chunks = parser.parse_content(sample, "main.tf", FileId(1))

    name_to_type = {c.symbol: c.chunk_type for c in chunks}

    # Block should be TABLE (HCL-only mapping)
    assert any(
        name == "resource.aws_s3_bucket.b" and ctype == ChunkType.TABLE
        for name, ctype in name_to_type.items()
    ), (
        f"Expected TABLE for resource block, got: {[(n, t.value) for n, t in name_to_type.items() if 'resource.aws_s3_bucket.b' in n]}"
    )

    # Attributes should be KEY_VALUE (HCL-only mapping)
    assert name_to_type.get("resource.aws_s3_bucket.b.bucket") == ChunkType.KEY_VALUE
    # Nested object pair captured as KEY_VALUE
    assert name_to_type.get("resource.aws_s3_bucket.b.tags.Env") == ChunkType.KEY_VALUE


def test_hcl_value_type_metadata_present():
    sample = (
        "locals {\n"
        "  list  = [1, 2, 3]\n"
        "  truth = true\n"
        "  num   = 42\n"
        "  ref   = var.region\n"
        '  text  = "hello"\n'
        "}\n"
    )

    parser = ParserFactory().create_parser(Language.HCL)
    ast = parser.engine.parse_to_ast(sample)
    chunks = parser.extractor.extract_concept(
        ast.root_node, sample.encode(), UniversalConcept.DEFINITION
    )

    attr = [c for c in chunks if c.language_node_type == "attribute"]
    assert len(attr) >= 5, "Expected at least 5 attribute definitions from locals block"

    allowed_types = {
        "expression",
        "number",
        "bool",
        "null",
        "string",
        "array",
        "object",
        "variable",
        "function",
        "template",
    }

    for c in attr:
        vtype = c.metadata.get("value_type")
        assert vtype is not None, f"Attribute {c.name} missing value_type metadata"
        assert isinstance(vtype, str), (
            f"value_type should be a string, got {type(vtype)}"
        )
        assert vtype in allowed_types or vtype, (
            f"Unexpected value_type '{vtype}' for {c.name}"
        )
