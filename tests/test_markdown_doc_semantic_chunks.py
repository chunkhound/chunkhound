from chunkhound.core.types.common import ChunkType, Language
from chunkhound.core.utils.embedding_utils import format_chunk_for_embedding
from chunkhound.parsers.parser_factory import create_parser_for_language


def test_markdown_sections_carry_frontmatter_and_heading_path_metadata():
    content = """---
type: decision
status: accepted
owner: docs
summary: Documentation knowledgebase contract
tags: [docs, agents]
related_requirements:
  - docs/requirements/REQ-agentic-doc-intelligence.md
---

# ADR-0001: Documentation knowledgebase contract

## Decision

Use typed Markdown frontmatter and explicit relationship fields.

## Consequences

Agents can retrieve docs by type, status, heading, and relationship.
"""

    parser = create_parser_for_language(Language.MARKDOWN)
    chunks = parser.parse_content(content)

    decision = next(chunk for chunk in chunks if chunk.symbol == "heading_decision")
    assert decision.chunk_type == ChunkType.HEADER_2
    assert decision.metadata["doc_type"] == "decision"
    assert decision.metadata["doc_status"] == "accepted"
    assert decision.metadata["doc_owner"] == "docs"
    assert decision.metadata["doc_tags"] == ["docs", "agents"]
    assert decision.metadata["heading_path"] == [
        "ADR-0001: Documentation knowledgebase contract",
        "Decision",
    ]
    assert decision.metadata["related_requirements"] == [
        "docs/requirements/REQ-agentic-doc-intelligence.md"
    ]


def test_doc_embedding_context_includes_frontmatter_heading_and_relationships():
    text = format_chunk_for_embedding(
        code="## Decision\n\nUse typed Markdown frontmatter.",
        file_path="docs/decisions/ADR-0001.md",
        language="markdown",
        doc_metadata={
            "doc_title": "ADR-0001: Documentation knowledgebase contract",
            "doc_type": "decision",
            "doc_status": "accepted",
            "doc_owner": "communityx",
            "doc_summary": "Decision to model docs as a typed knowledgebase.",
            "doc_tags": ["docs", "agents"],
            "heading_path": [
                "ADR-0001: Documentation knowledgebase contract",
                "Decision",
            ],
            "relationship_keys": ["related_requirements"],
            "related_requirements": ["docs/requirements/REQ-agentic-doc-intelligence.md"],
        },
    )

    assert "type: decision" in text
    assert "status: accepted" in text
    assert "heading_path: ADR-0001: Documentation knowledgebase contract > Decision" in text
    assert "related_requirements: docs/requirements/REQ-agentic-doc-intelligence.md" in text
    assert text.endswith("Use typed Markdown frontmatter.")
