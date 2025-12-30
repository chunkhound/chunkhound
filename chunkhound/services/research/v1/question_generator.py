"""Question Generator for Deep Research Service.

This module handles all aspects of follow-up question generation and management:
1. LLM-powered question generation based on explored code and context
2. Question synthesis - combining multiple questions into high-level research directions
3. Relevance filtering - ensuring questions remain aligned with root research goals

Strategy:
- Generate questions targeting NEW code elements not yet explored
- Use exploration gist to prevent redundant paths
- Synthesize when level has too many questions (compress while maintaining coverage)
- Filter by relevance to root query and architectural value
- Apply adaptive token budgets based on traversal depth
"""

from typing import Any

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.research.shared.chunk_context_builder import (
    ChunkContextBuilder,
)
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.models import (
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    MAX_FOLLOWUP_QUESTIONS,
    QUESTION_FILTERING_TOKENS,
    QUESTION_SYNTHESIS_TOKENS,
    BFSNode,
    ResearchContext,
)


class QuestionGenerator:
    """Generates and manages follow-up questions for deep research exploration."""

    def __init__(
        self,
        llm_manager: LLMManager,
        import_context_service: ImportContextService | None = None,
    ):
        """Initialize question generator.

        Args:
            llm_manager: LLM manager for generating questions and synthesis
            import_context_service: Optional service for extracting imports from chunks.
                If None, chunk-based context building will skip import headers.
        """
        self._llm_manager = llm_manager
        self._import_context_service = import_context_service
        self._node_counter = 0

    def set_node_counter(self, counter: int) -> None:
        """Set the node counter for ID generation.

        Args:
            counter: Current node counter value
        """
        self._node_counter = counter

    def _get_next_node_id(self) -> int:
        """Get next node ID for graph traversal."""
        self._node_counter += 1
        return self._node_counter

    async def generate_follow_up_questions(
        self,
        query: str,
        context: ResearchContext,
        file_contents: dict[str, str],
        chunks: list[dict[str, Any]],
        global_explored_data: dict[str, Any],
        exploration_gist: str | None,
        max_input_tokens: int | None = None,
        depth: int = 0,
        max_depth: int = 1,
        constants_context: str = "",
    ) -> list[str]:
        """Generate follow-up questions using LLM.

        Args:
            query: Current query
            context: Research context
            file_contents: File contents found
            chunks: Chunks found
            global_explored_data: Global state with all explored chunks/files
            exploration_gist: Pre-built exploration gist (for efficiency)
            max_input_tokens: Maximum tokens for LLM input (uses adaptive budget if provided)
            depth: Current depth in BFS traversal
            max_depth: Maximum depth for this codebase
            constants_context: Constants ledger context for follow-up generation

        Returns:
            List of follow-up questions
        """
        # Validate that max_input_tokens was provided by caller
        if max_input_tokens is None:
            raise ValueError(
                "max_input_tokens must be provided by caller. "
                "Use adaptive budget system to calculate appropriate value."
            )

        llm = self._llm_manager.get_utility_provider()

        # Define JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Array of 0-{MAX_FOLLOWUP_QUESTIONS} follow-up questions",
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        }

        # Simplified system prompt per GPT-5-Nano best practices
        system = prompts.FOLLOWUP_GENERATION_SYSTEM

        # Build code context - from file_contents if available, otherwise from chunks
        max_tokens_for_context = max_input_tokens

        if file_contents:
            # Use file-based context (backwards compatible path)
            code_context = []
            total_tokens = 0

            for path, content in file_contents.items():
                content_tokens = llm.estimate_tokens(content)
                if total_tokens + content_tokens <= max_tokens_for_context:
                    code_context.append(
                        f"File: {path}\n{'=' * 60}\n{content}\n{'=' * 60}"
                    )
                    total_tokens += content_tokens
                else:
                    # Truncate to fit within total budget
                    remaining_tokens = max_tokens_for_context - total_tokens
                    if remaining_tokens > 500:  # Only include if meaningful
                        chars_to_include = remaining_tokens * 4  # ~4 chars per token
                        code_context.append(
                            f"File: {path}\n{'=' * 60}\n{content[:chars_to_include]}...\n{'=' * 60}"
                        )
                    break

            logger.debug(
                f"LLM input: {total_tokens} tokens from {len(code_context)} files "
                f"(budget: {max_tokens_for_context})"
            )
            code_section = (
                "\n\n".join(code_context) if code_context else "No code files loaded"
            )
        elif chunks:
            # Build context from chunks when file_contents not available
            builder = ChunkContextBuilder(
                import_context_service=self._import_context_service,
                llm_manager=self._llm_manager,
            )
            code_section = builder.build_code_context_with_imports(
                chunks, max_tokens_for_context
            )
            if not code_section:
                code_section = "No code files loaded"
            logger.debug(
                f"LLM input: built from {len(chunks)} chunks "
                f"(budget: {max_tokens_for_context})"
            )
        else:
            # Neither file_contents nor chunks available
            logger.warning(
                f"Cannot generate follow-up questions: no file contents or chunks. "
                f"Query: {query}"
            )
            return []

        # Also include chunk snippets for context
        chunks_preview = "\n".join(
            [
                f"- {chunk.get('file_path', 'unknown')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')} ({chunk.get('symbol', 'no symbol')})"
                for chunk in chunks[:10]
            ]
        )

        # Build common prompt sections
        gist_section = (
            f"""Already explored:
{exploration_gist}

"""
            if exploration_gist
            else ""
        )

        target_instruction = (
            "NEW code elements (not in explored files above)"
            if exploration_gist
            else "specific code elements found"
        )

        # Build constants section if available
        constants_section = (
            f"\nConstants:\n{constants_context}\n"
            if constants_context
            else ""
        )

        # Construct prompt with conditional gist section
        prompt = prompts.FOLLOWUP_GENERATION_USER.format(
            gist_section=gist_section,
            root_query=context.root_query,
            query=query,
            ancestors=" -> ".join(context.ancestors),
            code_section=code_section,
            chunks_preview=chunks_preview,
            constants_section=constants_section,
            max_questions=MAX_FOLLOWUP_QUESTIONS,
            target_instruction=target_instruction,
        )

        # Calculate adaptive output budget (scales 3k → 8k with depth)
        depth_ratio = depth / max(max_depth, 1)
        max_output_tokens = int(
            FOLLOWUP_OUTPUT_TOKENS_MIN
            + (FOLLOWUP_OUTPUT_TOKENS_MAX - FOLLOWUP_OUTPUT_TOKENS_MIN) * depth_ratio
        )
        logger.debug(
            f"Follow-up generation budget: {max_output_tokens:,} tokens "
            f"(depth {depth}/{max_depth}, ratio {depth_ratio:.2f})"
        )

        try:
            result = await llm.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=max_output_tokens,
            )

            questions = result.get("questions", [])

            # Filter empty questions
            valid_questions = [q.strip() for q in questions if q and q.strip()]

            # Filter questions by relevance to root query
            if valid_questions:
                valid_questions = await self.filter_relevant_followups(
                    valid_questions, context.root_query, query, context
                )

            return valid_questions[:MAX_FOLLOWUP_QUESTIONS]

        except Exception as e:
            logger.warning(
                f"Follow-up question generation failed: {e}, returning empty list"
            )
            return []

    async def synthesize_questions(
        self, nodes: list[BFSNode], context: ResearchContext, target_count: int
    ) -> list[BFSNode]:
        """Synthesize N new questions capturing unexplored aspects of input questions.

        Purpose: When BFS level has too many questions, synthesize them into
        fewer high-level questions that explore NEW areas for comprehensive coverage.

        Args:
            nodes: List of BFS nodes to synthesize
            context: Research context
            target_count: Target number of synthesized questions

        Returns:
            Fresh BFSNode objects with synthesized queries and empty metadata.
            These nodes will find their own chunks during processing.
        """
        if len(nodes) <= target_count:
            return nodes

        # Quality pre-filtering: Remove low-quality questions before synthesis
        filtered_nodes = [
            node
            for node in nodes
            if len(node.query.strip()) > 10  # Minimum length
            and not node.query.lower()
            .strip()
            .startswith(("what is ", "is there ", "does "))  # Avoid simple yes/no
        ]

        # If filtering reduced below target, skip synthesis
        if len(filtered_nodes) <= target_count:
            logger.debug(
                f"After quality filtering, {len(filtered_nodes)} questions remain (<= target {target_count}), skipping synthesis"
            )
            return filtered_nodes

        # Use filtered nodes for synthesis
        synthesis_nodes = filtered_nodes
        questions_str = "\n".join(
            [f"{i + 1}. {node.query}" for i, node in enumerate(synthesis_nodes)]
        )

        llm = self._llm_manager.get_utility_provider()

        # Create synthetic merge parent
        merge_parent = BFSNode(
            query=f"[Merge of {len(synthesis_nodes)} research directions]",
            depth=nodes[0].depth - 1,
            node_id=self._get_next_node_id(),
            children=synthesis_nodes,  # Reference all input nodes
        )

        # Define JSON schema with explanation parameter (forces reasoning)
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of synthesis strategy and why these questions explore different unexplored aspects",
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Array of 1 to {target_count} synthesized research questions, each exploring a distinct aspect",
                },
            },
            "required": ["reasoning", "questions"],
            "additionalProperties": False,
        }

        # Direct, unambiguous prompt optimized for GPT-5-Nano instruction adherence
        system = prompts.QUESTION_SYNTHESIS_SYSTEM

        prompt = prompts.QUESTION_SYNTHESIS_USER.format(
            root_query=context.root_query,
            questions_str=questions_str,
            target_count=target_count,
        )

        logger.debug(
            f"Question synthesis budget: {QUESTION_SYNTHESIS_TOKENS:,} tokens (model: {llm.model})"
        )

        try:
            result = await llm.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=QUESTION_SYNTHESIS_TOKENS,
            )

            reasoning = result.get("reasoning", "")
            synthesized_queries = result.get("questions", [])

            logger.debug(f"Synthesis reasoning: {reasoning}")

            # Validate that we got at least some questions
            if not synthesized_queries or len(synthesized_queries) == 0:
                logger.warning(
                    f"LLM returned empty questions array despite explicit requirement. "
                    f"Reasoning provided: '{reasoning}'. Falling back to truncated node list."
                )
                return synthesis_nodes[:target_count]

            # Create fresh BFSNode objects with empty metadata
            synthesized_nodes = []
            for query in synthesized_queries[:target_count]:
                if not query or not query.strip():
                    continue
                node = BFSNode(
                    query=query.strip(),
                    parent=merge_parent,  # Point to synthetic parent
                    depth=nodes[0].depth,  # Same depth as input nodes
                    node_id=self._get_next_node_id(),
                    chunks=[],  # Empty - will populate during processing
                    file_contents={},  # Empty - will populate during processing
                )
                synthesized_nodes.append(node)

            if not synthesized_nodes:
                logger.warning(
                    "All synthesized questions were empty, falling back to first N nodes"
                )
                return nodes[:target_count]

            logger.info(
                f"Synthesized {len(nodes)} questions into {len(synthesized_nodes)} new research directions"
            )

            return synthesized_nodes

        except Exception as e:
            logger.warning(
                f"Question synthesis failed: {e}, falling back to first N nodes"
            )
            return nodes[:target_count]

    async def filter_relevant_followups(
        self,
        questions: list[str],
        root_query: str,
        current_query: str,
        context: ResearchContext,
    ) -> list[str]:
        """Filter follow-ups by relevance to root query and architectural value.

        Args:
            questions: Candidate follow-up questions
            root_query: Original root query
            current_query: Current question being explored
            context: Research context

        Returns:
            Filtered list of most relevant follow-up questions
        """
        if len(questions) <= 1:
            return questions

        llm = self._llm_manager.get_utility_provider()

        questions_str = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

        system = prompts.QUESTION_FILTERING_SYSTEM

        prompt = prompts.QUESTION_FILTERING_USER.format(
            root_query=root_query,
            current_query=current_query,
            questions_str=questions_str,
            max_questions=MAX_FOLLOWUP_QUESTIONS,
        )

        logger.debug(
            f"Question filtering budget: {QUESTION_FILTERING_TOKENS:,} tokens (model: {llm.model})"
        )

        try:
            response = await llm.complete(
                prompt, system=system, max_completion_tokens=QUESTION_FILTERING_TOKENS
            )
            # Parse selected indices
            selected = [
                int(n.strip()) - 1
                for n in response.content.replace(",", " ").split()
                if n.strip().isdigit()
            ]
            filtered = [questions[i] for i in selected if 0 <= i < len(questions)]

            if filtered:
                logger.debug(
                    f"Filtered {len(questions)} follow-ups to {len(filtered)} relevant ones"
                )
                return filtered[:MAX_FOLLOWUP_QUESTIONS]
        except Exception as e:
            logger.warning(f"Follow-up filtering failed: {e}, using all questions")

        return questions[:MAX_FOLLOWUP_QUESTIONS]
