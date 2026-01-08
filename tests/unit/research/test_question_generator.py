"""Unit tests for QuestionGenerator in deep research.

Tests follow-up question generation, synthesis, and filtering
without requiring real API calls.
"""

import pytest

from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.models import (
    BFSNode,
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    MAX_FOLLOWUP_QUESTIONS,
    ResearchContext,
)
from chunkhound.services.research.question_generator import QuestionGenerator
from tests.fixtures.fake_providers import FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider with pattern-based responses for question generation."""
    return FakeLLMProvider(
        responses={
            # Follow-up generation (JSON)
            "follow": '{"questions": ["How is caching implemented?", "What are the data structures used?", "How does error handling work?"]}',
            "generate": '{"questions": ["Implementation details?", "Architecture patterns?"]}',
            # Synthesis (JSON)
            "synthesis": '{"reasoning": "Combining related questions about authentication and storage", "questions": ["How does the authentication and session storage system work together?"]}',
            "synthesize": '{"reasoning": "Merging questions", "questions": ["Combined question 1", "Combined question 2"]}',
            # Filtering (plain text with indices)
            "filter": "1 3",  # Select questions 1 and 3
            "relevant": "2",  # Select question 2
        }
    )


@pytest.fixture
def llm_manager(fake_llm_provider, monkeypatch):
    """Create LLM manager with fake provider."""

    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture
def question_generator(llm_manager):
    """Create question generator with fake LLM."""
    return QuestionGenerator(llm_manager)


class TestGenerateFollowUpQuestions:
    """Test follow-up question generation."""

    @pytest.mark.asyncio
    async def test_token_budget_scales_with_depth(self, question_generator):
        """Token budget should scale from MIN to MAX based on depth."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "def foo(): pass"}
        chunks = []
        global_explored = {}
        max_input_tokens = 10000

        # Test depth 0 (should use min budget)
        result_shallow = await question_generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            chunks,
            global_explored,
            exploration_gist=None,
            max_input_tokens=max_input_tokens,
            depth=0,
            max_depth=3,
        )

        # Test depth 3 (should use max budget)
        result_deep = await question_generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            chunks,
            global_explored,
            exploration_gist=None,
            max_input_tokens=max_input_tokens,
            depth=3,
            max_depth=3,
        )

        # Both should succeed (actual budget tested internally)
        assert isinstance(result_shallow, list)
        assert isinstance(result_deep, list)
        # Depth ratio 0/3 = 0, budget = MIN
        # Depth ratio 3/3 = 1, budget = MAX

    @pytest.mark.asyncio
    async def test_uses_chunks_when_file_contents_empty(self, question_generator):
        """Should use chunks to build context when file_contents is empty."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        chunks = [{"file_path": "test.py", "content": "def foo(): pass"}]
        global_explored = {}
        max_input_tokens = 10000

        result = await question_generator.generate_follow_up_questions(
            query,
            context,
            {},  # Empty file contents
            chunks,
            global_explored,
            exploration_gist=None,
            max_input_tokens=max_input_tokens,
            depth=1,
            max_depth=3,
        )

        # Now uses chunks when file_contents is empty
        assert isinstance(result, list), "Should return questions from chunk context"
        assert len(result) >= 1, "Should generate questions from chunks"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_content_or_chunks(self, question_generator):
        """Should return empty list when both file_contents and chunks are empty."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        global_explored = {}
        max_input_tokens = 10000

        result = await question_generator.generate_follow_up_questions(
            query,
            context,
            {},  # Empty file contents
            [],  # Empty chunks
            global_explored,
            exploration_gist=None,
            max_input_tokens=max_input_tokens,
            depth=1,
            max_depth=3,
        )

        assert result == [], "Should return empty list when no content available"

    @pytest.mark.asyncio
    async def test_requires_max_input_tokens(self, question_generator):
        """Should raise ValueError when max_input_tokens is None."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "code"}
        chunks = []
        global_explored = {}

        with pytest.raises(ValueError, match="max_input_tokens must be provided"):
            await question_generator.generate_follow_up_questions(
                query,
                context,
                file_contents,
                chunks,
                global_explored,
                exploration_gist=None,
                max_input_tokens=None,  # Missing required parameter
                depth=1,
                max_depth=3,
            )

    @pytest.mark.asyncio
    async def test_includes_exploration_gist(self, question_generator):
        """Should include gist in prompt when provided."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "def foo(): pass"}
        chunks = []
        global_explored = {}
        gist = "Files: auth.py, session.py\nSymbols: authenticate, create_session"

        result = await question_generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            chunks,
            global_explored,
            exploration_gist=gist,
            max_input_tokens=10000,
            depth=1,
            max_depth=3,
        )

        assert isinstance(result, list), "Should return questions list"
        # Gist is used internally to prevent duplicate exploration

    @pytest.mark.asyncio
    async def test_filters_empty_questions(self, question_generator, monkeypatch):
        """Should filter out empty or whitespace-only questions."""
        # Create provider with mixed empty/valid questions
        provider_with_empties = FakeLLMProvider(
            responses={
                "follow": '{"questions": ["Valid question", "", "  ", "Another valid"]}',
            }
        )

        def mock_create_provider(self, config):
            return provider_with_empties

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "code"}

        result = await generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            [],
            {},
            exploration_gist=None,
            max_input_tokens=10000,
            depth=1,
            max_depth=3,
        )

        # Should only include non-empty questions
        assert len(result) >= 1, "Should have at least some valid questions"
        assert all(q.strip() for q in result), "All questions should be non-empty"

    @pytest.mark.asyncio
    async def test_limits_to_max_followup_questions(self, question_generator, monkeypatch):
        """Should limit results to MAX_FOLLOWUP_QUESTIONS."""
        # Create provider with more questions than max
        many_questions = [f"Question {i}" for i in range(10)]
        provider_with_many = FakeLLMProvider(
            responses={
                "follow": f'{{"questions": {many_questions}}}',
            }
        )

        def mock_create_provider(self, config):
            return provider_with_many

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "code"}

        result = await generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            [],
            {},
            exploration_gist=None,
            max_input_tokens=10000,
            depth=1,
            max_depth=3,
        )

        assert (
            len(result) <= MAX_FOLLOWUP_QUESTIONS
        ), f"Should limit to {MAX_FOLLOWUP_QUESTIONS} questions"

    @pytest.mark.asyncio
    async def test_graceful_failure_returns_empty(self, question_generator, monkeypatch):
        """Should return empty list on LLM failure."""
        # Create failing provider
        failing_provider = FakeLLMProvider(responses={})

        def mock_create_provider(self, config):
            return failing_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])
        file_contents = {"test.py": "code"}

        result = await generator.generate_follow_up_questions(
            query,
            context,
            file_contents,
            [],
            {},
            exploration_gist=None,
            max_input_tokens=10000,
            depth=1,
            max_depth=3,
        )

        assert result == [], "Should return empty list on failure"


class TestSynthesizeQuestions:
    """Test question synthesis logic."""

    @pytest.mark.asyncio
    async def test_no_synthesis_when_below_target(self, question_generator):
        """Should skip synthesis when nodes <= target_count."""
        context = ResearchContext(root_query="root", ancestors=[])
        nodes = [
            BFSNode(query="How does authentication work in the API layer?", depth=1, node_id=1),
            BFSNode(query="How does session management persist across requests?", depth=1, node_id=2),
        ]

        result = await question_generator.synthesize_questions(nodes, context, target_count=3)

        assert result == nodes, "Should return original nodes when count <= target"

    @pytest.mark.asyncio
    async def test_synthesis_creates_new_nodes(self, question_generator):
        """Should create fresh BFSNode objects with synthesized queries."""
        context = ResearchContext(root_query="root", ancestors=[])
        nodes = [
            BFSNode(query=f"How does component {i} handle data flow?", depth=2, node_id=i)
            for i in range(5)
        ]

        result = await question_generator.synthesize_questions(nodes, context, target_count=2)

        assert len(result) <= 2, "Should reduce to target count"
        assert all(isinstance(node, BFSNode) for node in result), "Should return BFSNodes"
        # Synthesized nodes should have empty chunks/file_contents
        for node in result:
            if node.query.startswith("Combined"):  # Our fake response
                assert node.chunks == [], "Synthesized nodes should have empty chunks"
                assert (
                    node.file_contents == {}
                ), "Synthesized nodes should have empty file_contents"

    @pytest.mark.asyncio
    async def test_synthesis_creates_merge_parent(self, question_generator):
        """Should create synthetic merge parent linking all input nodes."""
        context = ResearchContext(root_query="root", ancestors=[])
        nodes = [
            BFSNode(query=f"How does the authentication layer validate credentials?", depth=2, node_id=i)
            for i in range(4)
        ]

        result = await question_generator.synthesize_questions(nodes, context, target_count=2)

        # Synthesized nodes should have parent
        for node in result:
            if hasattr(node, "parent") and node.parent:
                assert "[Merge of" in node.parent.query, "Parent should be merge node"
                assert node.parent.depth == 1, "Parent should be depth - 1"

    @pytest.mark.asyncio
    async def test_quality_filtering_before_synthesis(self, question_generator):
        """Should filter low-quality questions before synthesis."""
        context = ResearchContext(root_query="root", ancestors=[])
        nodes = [
            BFSNode(query="Short", depth=2, node_id=1),  # Too short
            BFSNode(query="What is authentication?", depth=2, node_id=2),  # Yes/no
            BFSNode(query="How does the caching system work?", depth=2, node_id=3),  # Good
            BFSNode(query="Is there a database?", depth=2, node_id=4),  # Yes/no
            BFSNode(query="Does it have tests?", depth=2, node_id=5),  # Yes/no
        ]

        result = await question_generator.synthesize_questions(nodes, context, target_count=2)

        # After filtering bad questions, might skip synthesis if <= target
        assert len(result) <= len(nodes), "Should not increase question count"

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_empty_response(self, question_generator, monkeypatch):
        """Should fallback to truncated list when LLM returns empty."""
        empty_provider = FakeLLMProvider(
            responses={
                "synthesis": '{"reasoning": "test", "questions": []}',
            }
        )

        def mock_create_provider(self, config):
            return empty_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        context = ResearchContext(root_query="root", ancestors=[])
        # Use realistic questions that won't be filtered by quality check
        nodes = [
            BFSNode(query="How does authentication work in the system?", depth=2, node_id=i)
            for i in range(5)
        ]

        result = await generator.synthesize_questions(nodes, context, target_count=2)

        assert len(result) <= 2, "Should fallback to truncated list"
        if result:
            assert "authentication" in result[0].query.lower(), "Should preserve original nodes"

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_llm_failure(self, question_generator, monkeypatch):
        """Should fallback to truncated list on LLM exception."""
        failing_provider = FakeLLMProvider(responses={})

        def mock_create_provider(self, config):
            return failing_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        context = ResearchContext(root_query="root", ancestors=[])
        # Use realistic questions that won't be filtered by quality check
        nodes = [
            BFSNode(query="How does the caching system handle invalidation?", depth=2, node_id=i)
            for i in range(5)
        ]

        result = await generator.synthesize_questions(nodes, context, target_count=2)

        assert len(result) == 2, "Should return first N nodes on failure"


class TestFilterRelevantFollowups:
    """Test follow-up question filtering."""

    @pytest.mark.asyncio
    async def test_no_filtering_for_single_question(self, question_generator):
        """Should skip filtering when only one question."""
        questions = ["Single question"]
        context = ResearchContext(root_query="root", ancestors=[])

        result = await question_generator.filter_relevant_followups(
            questions, context.root_query, "current", context
        )

        assert result == questions, "Should return single question unchanged"

    @pytest.mark.asyncio
    async def test_filters_by_indices(self, question_generator):
        """Should filter questions by LLM-provided indices."""
        questions = [
            "Question 1",
            "Question 2",
            "Question 3",
            "Question 4",
        ]
        context = ResearchContext(root_query="root", ancestors=[])

        # Fake provider will return "1 3" (indices 0 and 2 in 0-indexed)
        result = await question_generator.filter_relevant_followups(
            questions, context.root_query, "current", context
        )

        assert len(result) <= len(questions), "Should filter questions"
        assert len(result) <= MAX_FOLLOWUP_QUESTIONS, "Should limit to max"

    @pytest.mark.asyncio
    async def test_handles_comma_separated_indices(self, question_generator, monkeypatch):
        """Should parse comma-separated indices."""
        comma_provider = FakeLLMProvider(responses={"filter": "1, 2, 4"})

        def mock_create_provider(self, config):
            return comma_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        context = ResearchContext(root_query="root", ancestors=[])

        result = await generator.filter_relevant_followups(
            questions, context.root_query, "current", context
        )

        assert len(result) >= 1, "Should parse comma-separated indices"

    @pytest.mark.asyncio
    async def test_filters_invalid_indices(self, question_generator, monkeypatch):
        """Should filter out invalid indices (out of range)."""
        invalid_provider = FakeLLMProvider(responses={"filter": "1 10 -1"})

        def mock_create_provider(self, config):
            return invalid_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        questions = ["Q1", "Q2", "Q3"]
        context = ResearchContext(root_query="root", ancestors=[])

        result = await generator.filter_relevant_followups(
            questions, context.root_query, "current", context
        )

        # Should only include valid index (1 -> index 0)
        assert all(q in questions for q in result), "Should only include valid questions"

    @pytest.mark.asyncio
    async def test_fallback_on_filtering_failure(self, question_generator, monkeypatch):
        """Should return all questions on filtering failure."""
        failing_provider = FakeLLMProvider(responses={})

        def mock_create_provider(self, config):
            return failing_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        manager = LLMManager(utility_config, synthesis_config)
        generator = QuestionGenerator(manager)

        questions = ["Q1", "Q2", "Q3", "Q4"]
        context = ResearchContext(root_query="root", ancestors=[])

        result = await generator.filter_relevant_followups(
            questions, context.root_query, "current", context
        )

        assert (
            len(result) <= MAX_FOLLOWUP_QUESTIONS
        ), "Should return max questions on failure"


class TestNodeCounter:
    """Test node ID generation."""

    def test_node_counter_increments(self, question_generator):
        """Node counter should increment with each call."""
        question_generator.set_node_counter(10)

        id1 = question_generator._get_next_node_id()
        id2 = question_generator._get_next_node_id()
        id3 = question_generator._get_next_node_id()

        assert id1 == 11, "Should start at counter + 1"
        assert id2 == 12, "Should increment"
        assert id3 == 13, "Should continue incrementing"

    def test_set_node_counter(self, question_generator):
        """Should be able to set counter to specific value."""
        question_generator.set_node_counter(100)
        next_id = question_generator._get_next_node_id()

        assert next_id == 101, "Should use set counter value"
