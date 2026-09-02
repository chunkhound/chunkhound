"""Result enhancement utilities for search operations.

This module provides functionality for enriching and combining search results
with additional metadata, formatting, and scoring logic.
"""

import re
from pathlib import Path
from typing import Any


def _strip_chunk_part_suffix(symbol: str) -> str:
    """Strip _partN suffixes from symbol names added during chunk splitting.

    During chunk splitting, symbols get suffixes like:
    - Binary split: _part1, _part2
    - Multi-part: _part3, _part4, ..., _partN
    - Nested: _part1_part2, _part2_part1, etc.

    These suffixes are internal chunking artifacts that don't exist in actual
    source code. This function removes them to expose the original code identifier.

    Args:
        symbol: Symbol name potentially containing _partN suffixes

    Returns:
        Clean symbol name without chunk suffixes

    Examples:
        >>> _strip_chunk_part_suffix("DeepResearchService_part2")
        'DeepResearchService'
        >>> _strip_chunk_part_suffix("MyClass_part1_part2")
        'MyClass'
        >>> _strip_chunk_part_suffix("block_line_767_part3")
        'block_line_767'
        >>> _strip_chunk_part_suffix("normal_function")
        'normal_function'
    """
    # Pattern matches _partN suffixes (including nested ones)
    # (?:_part\d+)+ matches one or more occurrences of _part followed by digits
    # $ ensures we only strip from the end
    return re.sub(r"(?:_part\d+)+$", "", symbol)


class ResultEnhancer:
    """Utility class for enhancing and combining search results."""

    def enhance_search_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Enhance search result with additional metadata and formatting.

        Args:
            result: Raw search result from database

        Returns:
            Enhanced result with additional metadata
        """
        enhanced = result.copy()

        # Clean symbol name by stripping _partN suffixes added during chunk splitting
        # This ensures consumers (like deep research) get valid code identifiers
        # instead of internal chunking artifacts
        if "symbol" in result and result["symbol"]:
            original_symbol = result["symbol"]
            clean_symbol = _strip_chunk_part_suffix(original_symbol)

            # Only modify if suffix was actually stripped
            if clean_symbol != original_symbol:
                enhanced["symbol"] = clean_symbol

                # Preserve original symbol in metadata for debugging/compatibility
                metadata = enhanced.get("metadata", {})
                if isinstance(metadata, dict):
                    metadata["original_symbol"] = original_symbol
                    enhanced["metadata"] = metadata
                else:
                    # If metadata isn't a dict, create a new metadata dict
                    enhanced["metadata"] = {"original_symbol": original_symbol}

        # Add computed fields
        if "start_line" in result and "end_line" in result:
            enhanced["line_count"] = result["end_line"] - result["start_line"] + 1

        # Add code preview (truncated if too long)
        if "code" in result and result["code"]:
            code = result["code"]
            if len(code) > 500:
                enhanced["code_preview"] = code[:500] + "..."
                enhanced["is_truncated"] = True
            else:
                enhanced["code_preview"] = code
                enhanced["is_truncated"] = False

        # Add file extension for quick language identification
        if "path" in result:
            file_path = result["path"]
            enhanced["file_extension"] = Path(file_path).suffix.lower()

        # Format similarity score if present
        if "similarity" in result:
            enhanced["similarity_percentage"] = round(result["similarity"] * 100, 2)

        return enhanced

    def combine_search_results(
        self,
        semantic_results: list[dict[str, Any]],
        regex_results: list[dict[str, Any]],
        semantic_weight: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Combine semantic and regex search results with weighted ranking.

        Args:
            semantic_results: Results from semantic search
            regex_results: Results from regex search
            semantic_weight: Weight for semantic results (0.0-1.0)
            limit: Maximum number of results to return

        Returns:
            Combined and ranked results
        """
        combined = {}
        regex_weight = 1.0 - semantic_weight

        # Process semantic results
        for i, result in enumerate(semantic_results):
            chunk_id = result.get("chunk_id") or result.get("id")
            if chunk_id:
                # Score based on position and similarity
                position_score = (len(semantic_results) - i) / len(semantic_results)
                similarity_score = result.get("similarity", 0.5)
                score = (
                    position_score * 0.3 + similarity_score * 0.7
                ) * semantic_weight

                combined[chunk_id] = {
                    **result,
                    "search_type": "semantic",
                    "combined_score": score,
                    "semantic_score": similarity_score,
                }

        # Process regex results
        for i, result in enumerate(regex_results):
            chunk_id = result.get("chunk_id") or result.get("id")
            if chunk_id:
                # Score based on position (regex has no similarity score)
                position_score = (len(regex_results) - i) / len(regex_results)
                score = position_score * regex_weight

                if chunk_id in combined:
                    # Boost existing result
                    combined[chunk_id]["combined_score"] += score
                    combined[chunk_id]["search_type"] = "hybrid"
                    combined[chunk_id]["regex_score"] = position_score
                else:
                    combined[chunk_id] = {
                        **result,
                        "search_type": "regex",
                        "combined_score": score,
                        "regex_score": position_score,
                    }

        # Sort by combined score and return top results
        sorted_results = sorted(
            combined.values(), key=lambda x: x["combined_score"], reverse=True
        )

        return sorted_results[:limit]
