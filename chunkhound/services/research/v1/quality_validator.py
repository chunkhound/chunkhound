"""Quality validation for deep research outputs.

This module provides quality control mechanisms for synthesis outputs including:
- Verbosity filtering: Remove common LLM meta-commentary and defensive patterns
- Quality validation: Check for theoretical placeholders, citation density, and vague measurements
- Citation validation: Ensure proper numbered reference citations
- Citation sorting: Sort consecutive citation sequences for readability

Quality checks act as safety nets even with good prompting, ensuring outputs are
concrete, well-cited, and actionable rather than vague or theoretical.
"""

import re
from typing import Any

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.models import (
    _CITATION_PATTERN,
    _CITATION_SEQUENCE_PATTERN,
    REQUIRE_CITATIONS,
)


class QualityValidator:
    """Quality validation for deep research synthesis outputs."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize quality validator.

        Args:
            llm_manager: LLM manager for token estimation
        """
        self._llm_manager = llm_manager

    def filter_verbosity(self, text: str) -> str:
        """Remove common LLM verbosity patterns from synthesis output.

        Acts as safety net even with good prompts. Strips defensive caveats,
        meta-commentary, and unnecessary qualifications.

        Args:
            text: Synthesis text to filter

        Returns:
            Filtered text with verbose patterns removed
        """
        # Patterns to remove (from research on LLM verbosity)
        patterns_to_remove = [
            r"It'?s important to note that\s+",
            r"It'?s worth noting that\s+",
            r"It should be noted that\s+",
            r"However, it should be mentioned that\s+",
            r"Please note that\s+",
            r"As mentioned (?:earlier|above|previously),?\s+",
            # Remove standalone "No information found" lines from body (keep in "Missing:" context)
            r"^No information (?:was )?found (?:for|about)[^\n]+\n",
            r"^Unfortunately, the (?:code|analysis) does not (?:show|provide)[^\n]+\n",
            # Remove vague precision statements
            r"The (?:exact|precise|specific) (?:implementation|details?|mechanism|values?) (?:is|are) not (?:provided|documented|shown|clear|available) in the (?:code|analysis)[,.]?\s*",
            r"(?:More|Additional) (?:research|investigation|analysis|context) (?:is|would be) (?:needed|required)[,.]?\s*",
        ]

        filtered = text
        for pattern in patterns_to_remove:
            filtered = re.sub(pattern, "", filtered, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive newlines left by removals (max 2 consecutive newlines)
        filtered = re.sub(r"\n{3,}", "\n\n", filtered)

        # Log if we actually filtered anything
        if filtered != text:
            chars_removed = len(text) - len(filtered)
            logger.debug(
                f"Verbosity filter removed {chars_removed} chars of meta-commentary"
            )

        return filtered

    def validate_output_quality(
        self, answer: str, target_tokens: int
    ) -> tuple[str, list[str]]:
        """Validate output quality for conciseness and actionability.

        Args:
            answer: Synthesized answer to validate
            target_tokens: Target token count for this output

        Returns:
            Tuple of (validated_answer, list_of_warnings)
        """
        warnings = []
        llm = self._llm_manager.get_utility_provider()

        # Check 1: Detect theoretical placeholders
        theoretical_patterns = [
            "provide exact",
            "provide precise",
            "specify exact",
            "implementation-dependent",
            "precise line-level mappings",
            "exact numeric budgets",
            "provide the actual",
            "should specify",
            "need to determine",
            "requires clarification",
        ]

        for pattern in theoretical_patterns:
            if pattern.lower() in answer.lower():
                warnings.append(
                    f"QUALITY: Output contains theoretical placeholder: '{pattern}'. "
                    "This suggests lack of concrete information."
                )
                logger.warning(f"Output quality issue: contains '{pattern}'")

        # Check 2: Citation density (should have reasonable citations)
        citations = _CITATION_PATTERN.findall(answer)
        citation_count = len(citations)
        answer_tokens = llm.estimate_tokens(answer)

        if answer_tokens > 1000 and citation_count < 5:
            warnings.append(
                f"QUALITY: Low citation density ({citation_count} citations in {answer_tokens} tokens). "
                "Output may lack concrete code references."
            )
            logger.warning(
                f"Low citation density: {citation_count} citations in {answer_tokens} tokens"
            )

        # Check 3: Excessive length
        if answer_tokens > target_tokens * 1.5:
            warnings.append(
                f"QUALITY: Output is verbose ({answer_tokens:,} tokens vs {target_tokens:,} target). "
                "May need tighter prompting."
            )
            logger.warning(
                f"Verbose output: {answer_tokens:,} tokens (target: {target_tokens:,})"
            )

        # Check 4: Vague measurements (should use exact numbers)
        vague_patterns = [
            r"\b(several|many|few|some|various|multiple|numerous)\s+(seconds|minutes|items|entries|elements|chunks)",
            r"\b(around|approximately|roughly|about)\s+\d+",
            r"\bhundreds of\b",
            r"\bthousands of\b",
        ]

        for pattern in vague_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"QUALITY: Vague measurement detected: {matches[0]}. "
                    "Should use exact values."
                )
                logger.warning(f"Vague measurement in output: {matches[0]}")
                break  # Only report first instance

        return answer, warnings

    def validate_citations(self, answer: str, chunks: list[dict[str, Any]]) -> str:
        """Ensure answer contains numbered reference citations.

        Args:
            answer: Answer text to validate
            chunks: Chunks that were analyzed

        Returns:
            Answer with citations appended if missing
        """
        if not REQUIRE_CITATIONS:
            return answer

        # Check for citation pattern: [N] where N is a reference number
        citations = _CITATION_PATTERN.findall(answer)

        answer_length = len(answer.strip())
        answer_lines = answer.count("\n") + 1
        citation_count = len(citations)

        # Calculate citation density (citations per 100 lines of analysis)
        citation_density = (
            (citation_count / answer_lines * 100) if answer_lines > 0 else 0
        )

        # Log citation metrics
        logger.info(
            f"Citation metrics: {citation_count} citations found in {answer_length:,} chars "
            f"({answer_lines} lines), density={citation_density:.1f} citations/100 lines"
        )

        # Sample citations for debugging (show first 3)
        if citations:
            sample_citations = citations[:3]
            logger.debug(f"Sample citations: {', '.join(sample_citations)}")

        if not citations and chunks:
            # Answer missing inline citations - footer provides separate comprehensive listing
            # Enhanced warning with context
            if answer_length == 0:
                logger.warning(
                    "LLM answer is EMPTY - this indicates an LLM error. "
                    "Should have been caught by synthesis validation."
                )
            elif answer_length < 200:
                logger.warning(
                    f"LLM answer suspiciously short ({answer_length} chars) and missing "
                    f"reference citations [N] in analysis body"
                )
            else:
                logger.warning(
                    f"LLM answer missing reference citations [N] in analysis body "
                    f"(answer_length={answer_length} chars, {answer_lines} lines). "
                    f"Check if prompt citation examples are being followed."
                )
        elif citations and citation_density < 1.0:
            # Has some citations but density is low
            logger.warning(
                f"Low citation density: {citation_density:.1f} citations/100 lines "
                f"({citation_count} total citations). Consider reviewing prompt emphasis."
            )

        # Sort citation sequences for improved readability
        answer = self.sort_citation_sequences(answer)

        return answer

    def sort_citation_sequences(self, text: str) -> str:
        """Sort inline citation sequences in ascending numerical order.

        Transforms sequences like [11][2][1][5] into [1][2][5][11] for improved
        readability. Only sorts consecutive citations - isolated citations and
        citations separated by text remain in their original positions.

        Args:
            text: Text containing citation sequences

        Returns:
            Text with sorted citation sequences

        Examples:
            >>> sort_citation_sequences("Algorithm [11][2][1] uses BFS")
            "Algorithm [1][2][11] uses BFS"

            >>> sort_citation_sequences("Timeout [5] and threshold [3][1][2]")
            "Timeout [5] and threshold [1][2][3]"
        """

        def sort_sequence(match: re.Match[str]) -> str:
            """Extract numbers, sort numerically, and reconstruct."""
            sequence = match.group(0)
            numbers = re.findall(r"\d+", sequence)
            sorted_numbers = sorted(int(n) for n in numbers)
            return "".join(f"[{n}]" for n in sorted_numbers)

        return _CITATION_SEQUENCE_PATTERN.sub(sort_sequence, text)
