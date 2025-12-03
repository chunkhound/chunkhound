"""Shared extraction logic for JavaScript-family mappings (JS/TS/JSX/TSX).

Provides a single, minimal implementation of the universal extraction trio:
`extract_name`, `extract_metadata`, and `extract_content`.

All implementations are behavior-preserving copies of the existing JS/TS
methods, consolidated here to avoid duplication across mappings.
"""

from typing import TYPE_CHECKING, Any

from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode


class JSFamilyExtraction:
    """Mixin: shared extraction helpers for JS-family mappings."""

    # BaseMapping supplies get_node_text; subclasses must inherit it via MRO

    def extract_name(  # type: ignore[override]
        self,
        concept: "UniversalConcept",
        captures: dict[str, "TSNode"],
        content: bytes,
    ) -> str:
        source = content.decode("utf-8", errors="replace")

        if concept == UniversalConcept.DEFINITION:
            if "name" in captures:
                name_text = self.get_node_text(captures["name"], source).strip()  # type: ignore[attr-defined]
                return name_text or "definition"
            if "definition" in captures:
                node = captures["definition"]
                text = self.get_node_text(node, source).strip()  # type: ignore[attr-defined]
                if text.startswith("export default"):
                    return "export_default"
                if "module.exports" in text:
                    return "module_exports"
                line = node.start_point[0] + 1
                return f"definition_line_{line}"

        if concept == UniversalConcept.COMMENT and "definition" in captures:
            node = captures["definition"]
            return f"comment_line_{node.start_point[0] + 1}"

        return f"unnamed_{concept.value}"

    def extract_metadata(  # type: ignore[override]
        self,
        concept: "UniversalConcept",
        captures: dict[str, "TSNode"],
        content: bytes,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {"concept": concept.value}

        if concept == UniversalConcept.DEFINITION and "definition" in captures:
            node = captures["definition"]
            init = captures.get("init")
            target = init or node
            try:
                node_type = getattr(target, "type", "")
                if node_type == "object":
                    meta["chunk_type_hint"] = "object"
                elif node_type == "array":
                    meta["chunk_type_hint"] = "array"
                else:
                    for i in range(getattr(target, "child_count", 0)):
                        child = target.child(i)
                        if not child:
                            continue
                        if child.type == "object":
                            meta["chunk_type_hint"] = "object"
                            break
                        if child.type == "array":
                            meta["chunk_type_hint"] = "array"
                            break
            except Exception:
                # Best-effort only; do not set hint on failure
                pass
        return meta

    def extract_content(  # type: ignore[override]
        self,
        concept: "UniversalConcept",
        captures: dict[str, "TSNode"],
        content: bytes,
    ) -> str:
        source = content.decode("utf-8", errors="replace")
        node = captures.get("definition") or next(iter(captures.values()), None)
        return self.get_node_text(node, source) if node is not None else ""  # type: ignore[attr-defined]

