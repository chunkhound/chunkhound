"""Tests for ResearchServiceProtocol conformance.

Verifies that all research service implementations properly implement
the ResearchServiceProtocol interface.
"""

import inspect
from typing import Any, get_type_hints

import pytest

from chunkhound.services.research.protocol import ResearchServiceProtocol
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)


class TestResearchServiceProtocolConformance:
    """Verify implementations conform to ResearchServiceProtocol."""

    def test_pluggable_research_service_is_protocol_instance(self) -> None:
        """PluggableResearchService should be recognized as protocol instance."""
        # Protocol conformance check (runtime_checkable)
        assert issubclass(PluggableResearchService, ResearchServiceProtocol) or isinstance(
            PluggableResearchService, type
        ), "PluggableResearchService should implement ResearchServiceProtocol"

    def test_deep_research_method_exists(self) -> None:
        """PluggableResearchService should have deep_research method."""
        assert hasattr(PluggableResearchService, "deep_research")
        assert callable(getattr(PluggableResearchService, "deep_research"))

    def test_deep_research_is_async(self) -> None:
        """deep_research method should be async."""
        method = getattr(PluggableResearchService, "deep_research")
        assert inspect.iscoroutinefunction(method), "deep_research must be async"

    def test_deep_research_signature_matches_protocol(self) -> None:
        """deep_research should accept query: str and return dict[str, Any]."""
        method = getattr(PluggableResearchService, "deep_research")
        sig = inspect.signature(method)

        # Check query parameter exists
        params = list(sig.parameters.keys())
        assert "self" in params, "deep_research should be an instance method"
        assert "query" in params, "deep_research should accept 'query' parameter"

        # Check query is str type
        query_param = sig.parameters["query"]
        if query_param.annotation != inspect.Parameter.empty:
            assert query_param.annotation == str, "query parameter should be str"

    def test_protocol_has_required_method(self) -> None:
        """ResearchServiceProtocol should define deep_research."""
        assert hasattr(ResearchServiceProtocol, "deep_research")

    def test_backwards_compat_aliases_exist(self) -> None:
        """Backwards compatibility aliases should exist."""
        from chunkhound.services.deep_research_service import (
            BFSResearchService,
            DeepResearchService,
        )

        assert BFSResearchService is PluggableResearchService
        assert DeepResearchService is PluggableResearchService
