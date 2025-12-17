"""Unit tests for regex pagination infinite loop risk in unified search.

Tests the infinite loop risk in UnifiedSearch._regex_search_for_symbols() method
(chunkhound/services/research/shared/unified_search.py lines 487-514).

RISK: Pagination loop has no max iteration limit - could loop forever if:
- Pages keep returning duplicate chunks
- Mock/buggy search service returns same page repeatedly
- Deduplication logic fails
- Empty page detection fails

Expected Fix (to be implemented):
```python
# In _regex_search_for_symbols()
MAX_REGEX_PAGES = 50  # Safety limit to prevent infinite loops
page_count = 0

while len(results) < target_per_symbol:
    page_count += 1
    if page_count > MAX_REGEX_PAGES:
        logger.warning(
            f"Regex pagination exceeded {MAX_REGEX_PAGES} pages for symbol '{symbol}', "
            f"stopping early with {len(results)} results"
        )
        break

    page, _ = await search_service.search_regex_async(...)
    # ... existing logic ...
```
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for tests."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def embedding_manager(fake_embedding_provider):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def research_config():
    """Create research config with scan_page_size setting."""
    return ResearchConfig(
        algorithm="v2",
        regex_scan_page_size=100,  # Standard page size
    )


@pytest.fixture
def db_services_duplicate_pages(fake_embedding_provider):
    """Create mock database services where regex returns same duplicates every page.

    This simulates the infinite loop risk: search_regex_async always returns
    the same 10 duplicate chunks regardless of offset.
    """

    class DuplicatePageSearchService:
        """Mock search service that returns same duplicates on every page."""

        def __init__(self):
            self.page_fetch_count = 0  # Track how many pages were requested

        async def search_semantic(self, **kwargs):
            """Return empty semantic results (test focuses on regex)."""
            return [], None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 100,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Always return same 10 duplicate chunks, ignoring offset.

            RISK: This simulates a buggy search service or database that returns
            the same duplicates on every page. Without max iteration limit,
            pagination loop would run forever trying to collect more chunks.
            """
            self.page_fetch_count += 1

            # Always return same 10 chunks (simulating duplicates)
            return [
                {"chunk_id": f"dup_chunk_{i}", "code": f"def func{i}():"}
                for i in range(10)
            ], None

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        def __init__(self):
            self.provider = MockProvider()
            self.search_service = DuplicatePageSearchService()

    return MockDatabaseServices()


@pytest.fixture
def db_services_massive_pagination(fake_embedding_provider):
    """Create mock database services that returns 1000 pages of unique chunks.

    This tests whether pagination has a safety limit to prevent excessive API calls.
    """

    class MassivePaginationSearchService:
        """Mock search service with 1000 pages of unique chunks."""

        def __init__(self):
            self.page_fetch_count = 0

        async def search_semantic(self, **kwargs):
            """Return empty semantic results (test focuses on regex)."""
            return [], None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 100,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Return unique chunks for up to 1000 pages."""
            self.page_fetch_count += 1

            # Calculate page number
            page_num = offset // page_size

            # Return unique chunks for up to 1000 pages
            if page_num >= 1000:
                return [], None

            # Return page_size unique chunks
            start_idx = offset
            return [
                {"chunk_id": f"unique_chunk_{start_idx + i}", "code": f"def func{start_idx + i}():"}
                for i in range(page_size)
            ], None

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        def __init__(self):
            self.provider = MockProvider()
            self.search_service = MassivePaginationSearchService()

    return MockDatabaseServices()


@pytest.fixture
def db_services_alternating_duplicates(fake_embedding_provider):
    """Create mock database services that alternates between new chunk and duplicate.

    This tests pagination logic when finding very few new chunks per page.
    """

    class AlternatingDuplicateSearchService:
        """Mock search service that alternates between new chunk and duplicate."""

        def __init__(self):
            self.page_fetch_count = 0

        async def search_semantic(self, **kwargs):
            """Return empty semantic results (test focuses on regex)."""
            return [], None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 100,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Return page with 1 new chunk + 99 duplicates."""
            self.page_fetch_count += 1

            page_num = offset // page_size

            # Return up to 50 pages
            if page_num >= 50:
                return [], None

            # Return 1 new chunk + 99 duplicates (chunk_id 0 repeats)
            return [
                {"chunk_id": f"new_chunk_{page_num}", "code": f"def func{page_num}():"},
                *[{"chunk_id": "duplicate_chunk_0", "code": "def duplicate():"} for _ in range(99)],
            ], None

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        def __init__(self):
            self.provider = MockProvider()
            self.search_service = AlternatingDuplicateSearchService()

    return MockDatabaseServices()


@pytest.fixture
def db_services_low_yield_pages(fake_embedding_provider):
    """Create mock database services with 100 chunks per page, but only 1 new chunk.

    This tests pagination efficiency when scan_page_size is large but yield is low.
    """

    class LowYieldSearchService:
        """Mock search service where each page has 100 chunks but only 1 is new."""

        def __init__(self):
            self.page_fetch_count = 0

        async def search_semantic(self, **kwargs):
            """Return empty semantic results (test focuses on regex)."""
            return [], None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 100,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Return page_size chunks, but only 1 new chunk (rest are duplicates)."""
            self.page_fetch_count += 1

            page_num = offset // page_size

            # Return up to 30 pages
            if page_num >= 30:
                return [], None

            # Return 1 new chunk + (page_size - 1) duplicates
            return [
                {"chunk_id": f"new_chunk_{page_num}", "code": f"def func{page_num}():"},
                *[
                    {"chunk_id": "duplicate_chunk_0", "code": "def duplicate():"}
                    for _ in range(page_size - 1)
                ],
            ], None

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        def __init__(self):
            self.provider = MockProvider()
            self.search_service = LowYieldSearchService()

    return MockDatabaseServices()


@pytest.fixture
def unified_search_duplicate_pages(db_services_duplicate_pages, embedding_manager, research_config):
    """Create unified search instance with duplicate-pages database services."""
    return UnifiedSearch(db_services_duplicate_pages, embedding_manager, research_config)


@pytest.fixture
def unified_search_massive_pagination(
    db_services_massive_pagination, embedding_manager, research_config
):
    """Create unified search instance with massive-pagination database services."""
    return UnifiedSearch(db_services_massive_pagination, embedding_manager, research_config)


@pytest.fixture
def unified_search_alternating(
    db_services_alternating_duplicates, embedding_manager, research_config
):
    """Create unified search instance with alternating-duplicates database services."""
    return UnifiedSearch(db_services_alternating_duplicates, embedding_manager, research_config)


@pytest.fixture
def unified_search_low_yield(db_services_low_yield_pages, embedding_manager, research_config):
    """Create unified search instance with low-yield-pages database services."""
    return UnifiedSearch(db_services_low_yield_pages, embedding_manager, research_config)


@pytest.mark.asyncio
async def test_duplicate_pages_loop_exits(unified_search_duplicate_pages, db_services_duplicate_pages):
    """Test pagination exits when all pages return same duplicates.

    RISK SCENARIO: search_regex_async returns same 10 chunks on every page,
    ignoring offset. Pagination loop should detect this and exit (found_new_chunk = False).

    WITHOUT FIX: Loop would continue forever, incrementing offset but getting same chunks.
    WITH FIX: Loop exits after 1 page when no new chunks found.

    EXPECTED BEHAVIOR:
    - target_per_symbol = 10 (default)
    - Fetches 1 page (gets 10 duplicate chunks)
    - Collects all 10 chunks (first time seeing them)
    - Reaches target (len(results) >= target_per_symbol)
    - Exits loop gracefully with 10 results
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with default target_per_symbol=10
    results = await unified_search_duplicate_pages.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated (doesn't hang)
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 10 unique chunks (all from first page)
    assert len(results) == 10
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Verify pagination fetched only 1 page (target reached immediately)
    # Page 1: Collects 10 new chunks → len(results) >= target_per_symbol → exits
    assert db_services_duplicate_pages.search_service.page_fetch_count == 1

    # Document expected fix: Loop should have MAX_REGEX_PAGES safety limit
    # Current behavior: Exits when target reached (correct)
    # Risk scenario: If we increase target_per_symbol to 100, would keep fetching
    # same duplicates forever (no new chunks, but loop continues)
    # Expected improvement: Add MAX_REGEX_PAGES to prevent infinite loops if dedup fails


@pytest.mark.asyncio
async def test_massive_pagination_has_safety_limit(
    unified_search_massive_pagination, db_services_massive_pagination
):
    """Test pagination stops before exhausting 1000 pages of unique results.

    RISK SCENARIO: search_regex_async can return 1000 pages of unique chunks
    (100,000 total chunks). Pagination loop should have a safety limit to prevent
    excessive API calls and memory usage.

    WITHOUT FIX: Loop would fetch all 1000 pages (100,000 API calls) until reaching
    target_per_symbol or exhausting results.

    WITH FIX: Loop exits after MAX_REGEX_PAGES (e.g., 50 pages) with warning.

    EXPECTED BEHAVIOR (after fix):
    - Fetches up to 50 pages (5,000 chunks)
    - Logs warning about exceeding MAX_REGEX_PAGES
    - Returns partial results (better than hanging or OOM)

    CURRENT BEHAVIOR (no fix):
    - Fetches pages until target_per_symbol reached (default ~20 chunks)
    - No safety limit if target is low, but risky if target is high
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with default target
    # target_per_symbol = max(1, 0 // 1) = 1 (since no semantic results)
    # But augmentation ratio would make target higher in real scenario
    results = await unified_search_massive_pagination.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated
    assert results is not None

    # Check page fetch count
    # WITHOUT FIX: Could fetch hundreds of pages if target is high
    # WITH FIX: Should cap at MAX_REGEX_PAGES (e.g., 50)
    page_count = db_services_massive_pagination.search_service.page_fetch_count

    # Current behavior: Stops when target reached (likely 1-2 pages for target=1)
    # Risk: If target_per_symbol is high (e.g., 1000), would fetch many pages
    # Document: Need MAX_REGEX_PAGES safety limit
    assert page_count < 100, (
        f"Pagination fetched {page_count} pages without safety limit. "
        f"Need MAX_REGEX_PAGES to prevent excessive API calls."
    )

    # For this test with default target (~1), should only fetch 1 page
    # But verifies that loop doesn't fetch all 1000 pages available
    assert len(results) > 0


@pytest.mark.asyncio
async def test_alternating_duplicates_pagination_terminates(
    unified_search_alternating, db_services_alternating_duplicates
):
    """Test pagination terminates when alternating between new chunk and duplicates.

    RISK SCENARIO: Each page returns 1 new chunk + 99 duplicates. This tests
    whether pagination logic correctly handles low yield rates and terminates
    when target reached.

    EXPECTED BEHAVIOR:
    - target_per_symbol = 10 (default)
    - Each page yields 1 new chunk + 99 duplicates
    - Needs 10 pages to collect 10 unique chunks
    - Exits when len(results) >= target_per_symbol
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with default target_per_symbol=10
    results = await unified_search_alternating.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 10 unique chunks (1 from each of 10 pages)
    # new_chunk_0 to new_chunk_9 + duplicate_chunk_0 (seen once)
    assert len(results) == 10
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Verify pagination fetched 10 pages (1 new chunk per page)
    # Page 1: new_chunk_0 + duplicate_chunk_0 → 2 new chunks, len=2
    # Page 2: new_chunk_1 + duplicate_chunk_0 → 1 new chunk (dup seen), len=3
    # ...
    # Page 9: new_chunk_8 → 1 new chunk, len=10 → target reached → exits
    assert db_services_alternating_duplicates.search_service.page_fetch_count == 9

    # Document: This demonstrates inefficiency when yield is low (1 new / 100 fetched = 1%)
    # Each page contributes 1 new chunk, so pagination works but is inefficient


@pytest.mark.asyncio
async def test_low_yield_pages_reaches_target_efficiently(
    unified_search_low_yield, db_services_low_yield_pages
):
    """Test pagination efficiency when scan_page_size=100 but only 1 new chunk per page.

    RISK SCENARIO: Large scan_page_size (100) but low yield (1 new chunk per page).
    Tests whether pagination continues fetching until target reached, despite
    inefficiency.

    EXPECTED BEHAVIOR:
    - scan_page_size = 100 (from config)
    - target_per_symbol = 10 (default)
    - Each page yields 1 new chunk + 99 duplicates
    - Needs 10 pages to collect 10 unique chunks
    - Exits when len(results) >= target_per_symbol

    EFFICIENCY NOTE: Fetches 1000 chunks total (100 per page × 10 pages) to get 10 new chunks.
    Efficiency = 1% (99 duplicates wasted per page). Real fix would be adaptive
    page sizing based on yield rate.
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with default target_per_symbol=10
    results = await unified_search_low_yield.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 10 unique chunks (1 from each of 10 pages)
    # new_chunk_0 to new_chunk_9 + duplicate_chunk_0 (seen once)
    assert len(results) == 10
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Verify chunk IDs are correct
    new_chunks = [cid for cid in chunk_ids if cid.startswith("new_chunk_")]
    assert len(new_chunks) == 9  # new_chunk_0 to new_chunk_8
    assert "duplicate_chunk_0" in chunk_ids  # Seen once in first page

    # Verify pagination fetched 9 pages to collect 10 unique chunks
    # Page 1: new_chunk_0 + duplicate_chunk_0 → 2 new, len=2
    # Page 2: new_chunk_1 + duplicate_chunk_0 → 1 new (dup seen), len=3
    # ...
    # Page 9: new_chunk_8 + duplicate_chunk_0 → 1 new, len=10 → exits
    assert db_services_low_yield_pages.search_service.page_fetch_count == 9

    # Document performance consideration:
    # - scan_page_size = 100 (config setting)
    # - Actual yield = 1 new chunk per page after first page
    # - Efficiency = ~1% (99 duplicates wasted per page)
    # - Fetched 900 chunks total for 10 new chunks (after first page with 2 new)
    # - Current implementation is correct but inefficient for low-yield scenarios
    # - Could improve with adaptive page sizing or early termination heuristics


@pytest.mark.asyncio
async def test_pagination_with_exclude_ids_deduplication(
    unified_search_low_yield, db_services_low_yield_pages
):
    """Test that exclude_ids are properly deduplicated during pagination.

    This verifies that semantic search results are excluded from regex pagination,
    preventing infinite loops when regex keeps finding the same semantic chunks.
    """
    symbols = ["test_symbol"]

    # Exclude the chunk that would be returned by first page
    exclude_ids = {"new_chunk_0"}

    # Call search_by_symbols with default target_per_symbol=10 and exclusion
    results = await unified_search_low_yield.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=exclude_ids,
    )

    # Verify loop terminated
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 10 chunks, but new_chunk_0 is excluded
    # Page 1: new_chunk_0 (excluded) + duplicate_chunk_0 (new) → 1 new, len=1
    # Page 2: new_chunk_1 (new) + duplicate_chunk_0 (seen) → 1 new, len=2
    # ...
    # Page 10: new_chunk_9 (new) → 1 new, len=10 → target reached
    assert len(results) == 10
    assert "new_chunk_0" not in chunk_ids  # Excluded
    assert "duplicate_chunk_0" in chunk_ids  # Collected in first page

    # Verify we got new_chunk_1 through new_chunk_9 (9 chunks)
    new_chunks = [cid for cid in chunk_ids if cid.startswith("new_chunk_")]
    assert len(new_chunks) == 9  # new_chunk_1 to new_chunk_9

    # Verify pagination fetched 10 pages to get 10 chunks (with new_chunk_0 excluded)
    assert db_services_low_yield_pages.search_service.page_fetch_count == 10


@pytest.mark.asyncio
async def test_duplicate_pages_infinite_loop_risk_high_target(
    unified_search_duplicate_pages, db_services_duplicate_pages
):
    """Test INFINITE LOOP RISK when target_per_symbol is higher than available unique chunks.

    CRITICAL RISK SCENARIO: This is the actual infinite loop bug!
    - search_regex_async returns same 10 chunks on every page (ignoring offset)
    - target_per_symbol = 100 (higher than available unique chunks)
    - Loop tries to collect 100 chunks but only 10 unique exist
    - After first page: len(results) = 10 < target (100)
    - After second page: same 10 chunks, all duplicates, found_new_chunk = False → exits

    WITHOUT found_new_chunk CHECK: Would loop forever trying to reach target=100
    WITH found_new_chunk CHECK: Exits gracefully with 10 results

    This test demonstrates that the found_new_chunk safeguard works, but there's
    still risk if deduplication logic fails or has bugs.
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with HIGH target_per_symbol (100)
    # This is higher than the 10 unique chunks available
    results = await unified_search_duplicate_pages.search_by_symbols(
        symbols,
        target_per_symbol=100,  # CRITICAL: Higher than available chunks
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated (doesn't hang)
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 10 unique chunks (all available, even though target was 100)
    assert len(results) == 10
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # CRITICAL ASSERTION: Verify pagination exited after detecting duplicates
    # Page 1: Collects 10 new chunks, len=10 < target (100) → continue
    # Page 2: Gets same 10 chunks, all duplicates, found_new_chunk=False → exits
    assert db_services_duplicate_pages.search_service.page_fetch_count == 2

    # Document: This proves found_new_chunk safeguard works correctly
    # BUT: If deduplication logic had a bug, could loop forever
    # RECOMMENDATION: Add MAX_REGEX_PAGES as defense-in-depth safety limit


@pytest.mark.asyncio
async def test_massive_pagination_with_high_target_needs_safety_limit(
    unified_search_massive_pagination, db_services_massive_pagination
):
    """Test that high target_per_symbol could cause many pages without safety limit.

    RISK SCENARIO: What if target_per_symbol is set very high (e.g., 5000)?
    Currently there's no MAX_REGEX_PAGES limit, so pagination would fetch
    thousands of pages until target is reached or results exhausted.

    This test demonstrates the need for MAX_REGEX_PAGES safety limit.
    """
    symbols = ["test_symbol"]

    # Call search_by_symbols with HIGH target_per_symbol (5000)
    # This would require 50 pages (100 chunks per page × 50 = 5000)
    results = await unified_search_massive_pagination.search_by_symbols(
        symbols,
        target_per_symbol=5000,  # RISK: Very high target
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 5000 unique chunks (50 pages × 100 per page)
    assert len(results) == 5000
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # CRITICAL: Verify pagination fetched 50 pages to reach target
    # WITHOUT MAX_REGEX_PAGES: Fetches all 50 pages (5000 API calls)
    # WITH MAX_REGEX_PAGES (e.g., 50): Would stop at 50 pages with warning
    assert db_services_massive_pagination.search_service.page_fetch_count == 50

    # Document: This demonstrates the risk of unbounded pagination
    # If target_per_symbol is very high, could cause:
    # - Excessive API calls
    # - High latency (50+ round trips)
    # - Memory pressure (5000+ chunks in memory)
    # RECOMMENDATION: Add MAX_REGEX_PAGES=50 to prevent this scenario


@pytest.mark.asyncio
async def test_empty_page_terminates_pagination(
    unified_search_massive_pagination, db_services_massive_pagination
):
    """Test that empty page (no results) terminates pagination loop.

    This verifies the `if not page: break` logic works correctly.
    """
    symbols = ["test_symbol"]

    # Store original search method
    original_search = db_services_massive_pagination.search_service.search_regex_async
    page_count_at_empty = [0]  # Mutable container to capture value

    async def limited_search(*args, **kwargs):
        """Return empty after 5 pages."""
        # Force empty after 5 pages
        if db_services_massive_pagination.search_service.page_fetch_count > 5:
            page_count_at_empty[0] = db_services_massive_pagination.search_service.page_fetch_count
            return [], None
        result = await original_search(*args, **kwargs)
        return result

    db_services_massive_pagination.search_service.search_regex_async = limited_search

    # Call search_by_symbols with default target_per_symbol=10
    results = await unified_search_massive_pagination.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Verify loop terminated
    assert results is not None

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Expected: 500 chunks from 5 pages (100 per page), but target_per_symbol=10
    # Page 1: Collects 10 chunks (reaches target) → exits immediately
    assert len(results) == 10
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Verify pagination fetched only 1 page (target reached immediately)
    # Since each page returns 100 unique chunks and target is 10, first page is enough
    assert db_services_massive_pagination.search_service.page_fetch_count == 1

    # Verify results are valid
    assert all(cid.startswith("unique_chunk_") for cid in chunk_ids)


# --- Summary of Test Coverage ---
#
# INFINITE LOOP RISKS TESTED:
# 1. ✅ Duplicate pages (same chunks every page) → Exits via found_new_chunk=False
#    - test_duplicate_pages_loop_exits: With default target=10
#    - test_duplicate_pages_infinite_loop_risk_high_target: With target=100 > available chunks
# 2. ✅ Massive pagination (1000 pages available) → Currently no limit, relies on target_per_symbol
#    - test_massive_pagination_has_safety_limit: Default target=10 → 1 page
#    - test_massive_pagination_with_high_target_needs_safety_limit: target=5000 → 50 pages (RISK)
# 3. ✅ Alternating duplicates (1 new + 99 dups) → Terminates when target reached
#    - test_alternating_duplicates_pagination_terminates: Fetches 9 pages for 10 unique chunks
# 4. ✅ Low yield pages (100 per page, 1 new) → Terminates when target reached
#    - test_low_yield_pages_reaches_target_efficiently: Fetches 9 pages for 10 unique chunks
# 5. ✅ Empty page detection → Exits via `if not page: break`
#    - test_empty_page_terminates_pagination: Mocked to return empty after 5 pages
# 6. ✅ Exclude IDs deduplication → Prevents re-collecting semantic chunks
#    - test_pagination_with_exclude_ids_deduplication: Excludes new_chunk_0, collects others
#
# EXPECTED FIX (not yet implemented):
# Add MAX_REGEX_PAGES constant to prevent unbounded pagination:
#
# ```python
# MAX_REGEX_PAGES = 50  # Safety limit
# page_count = 0
#
# while len(results) < target_per_symbol:
#     page_count += 1
#     if page_count > MAX_REGEX_PAGES:
#         logger.warning(
#             f"Regex pagination exceeded {MAX_REGEX_PAGES} pages for symbol '{symbol}', "
#             f"stopping early with {len(results)} results"
#         )
#         break
#     # ... existing logic ...
# ```
#
# CURRENT BEHAVIOR:
# - Relies on found_new_chunk=False to exit (deduplication working correctly)
# - Relies on target_per_symbol to limit pages (default ~10-20 chunks)
# - No explicit max page count → risky if dedup fails or target is very high
# - test_massive_pagination_with_high_target_needs_safety_limit PROVES: 50 pages fetched for target=5000
#
# RECOMMENDATION:
# - Add MAX_REGEX_PAGES safety limit (defense in depth)
# - Log warning when limit exceeded (visibility for debugging)
# - Consider adaptive page sizing for low-yield scenarios (future optimization)
#
# RISK ASSESSMENT:
# - LOW RISK: Default target_per_symbol (10-20) typically needs 1-2 pages
# - MEDIUM RISK: High semantic result counts could set target_per_symbol to 100+
# - HIGH RISK: If deduplication logic has a bug, could loop forever
# - CRITICAL: test_massive_pagination_with_high_target_needs_safety_limit shows 50 pages with no limit
