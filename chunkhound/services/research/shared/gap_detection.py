"""Gap Detection Service for Research (Phase 2).

This module implements gap detection and filling for the research algorithm:
1. Cluster covered chunks using k-means
2. Shard clusters by token budget
3. Detect gaps via parallel LLM analysis (with ROOT query injection)
4. Embed and cluster gap queries
5. Unify similar gaps (with ROOT query injection)
6. Select gaps via elbow detection
7. Fill gaps via parallel unified search (independent, no shared state)
8. Global deduplication AFTER all fills complete
9. Merge with coverage chunks

Key Invariants:
- ROOT query must be in ALL LLM prompts
- Gap fills are independent (no shared mutable state)
- Global dedup ONLY after all fills complete
- Threshold floor: effective_threshold >= phase1_threshold
"""

import asyncio
import math
from typing import Any

import numpy as np
from loguru import logger
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
)

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.evidence_ledger import CONSTANTS_INSTRUCTION_SHORT
from chunkhound.services.research.shared.chunk_context_builder import (
    ChunkContextBuilder,
    get_chunk_text,
)
from chunkhound.services.research.shared.chunk_dedup import (
    deduplicate_chunks,
    merge_chunk_lists,
)
from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
    find_elbow_kneedle,
)
from chunkhound.services.research.shared.failure_tracker import FailureMetrics
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.models import (
    IMPORT_DEFAULT_SCORE,
    ResearchContext,
)
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.shared.gap_models import GapCandidate, UnifiedGap

# Token budget per gap detection cluster (affects LLM context size)
GAP_CLUSTER_TOKEN_BUDGET = 50_000

# Number of k-means initialization runs for reproducibility
KMEANS_N_INIT = 10


class GapDetectionService:
    """Service for detecting and filling semantic gaps in code coverage."""

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: EmbeddingManager,
        db_services: DatabaseServices,
        config: ResearchConfig,
        import_resolver: Any | None = None,
        import_context_service: ImportContextService | None = None,
    ):
        """Initialize gap detection service.

        Args:
            llm_manager: LLM manager for gap detection and unification
            embedding_manager: Embedding manager for gap query embeddings
            db_services: Database services bundle
            config: Research configuration
            import_resolver: Optional ImportResolverService for import resolution
            import_context_service: Optional ImportContextService for header injection
        """
        self._llm_manager = llm_manager
        self._embedding_manager = embedding_manager
        self._db_services = db_services
        self._config = config
        self._import_resolver = import_resolver
        self._import_context_service = import_context_service
        self._unified_search = UnifiedSearch(db_services, embedding_manager, config)

    async def detect_and_fill_gaps(
        self,
        root_query: str,
        covered_chunks: list[dict],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict], dict]:
        """Detect and fill semantic gaps in coverage (Phase 2 main entry point).

        Args:
            root_query: Original research query
            covered_chunks: Raw chunk dicts from Phase 1
            phase1_threshold: Quality threshold from Phase 1 (floor for gap results)
            path_filter: Path filter to pass through to gap filling
            constants_context: Constants ledger context for LLM prompts

        Returns:
            Tuple of (all_chunks, gap_stats) where:
                - all_chunks: Merged covered + gap chunks, deduplicated
                - gap_stats: Statistics about gap detection and filling
        """
        if not covered_chunks:
            logger.warning("No covered chunks to analyze for gaps")
            return covered_chunks, {
                "gaps_found": 0,
                "gaps_filled": 0,
                "chunks_added": 0,
            }

        logger.info(
            f"Phase 2: Gap detection starting with {len(covered_chunks)} covered chunks"
        )

        # Step 2.1: Cluster chunks with k-means
        cluster_groups = await self._cluster_chunks_kmeans(covered_chunks)
        logger.info(f"Step 2.1: Clustered into {len(cluster_groups)} semantic groups")

        # Step 2.2: Shard by token budget
        shards = self._shard_by_tokens(cluster_groups)
        logger.info(f"Step 2.2: Created {len(shards)} shards from clusters")

        # Step 2.3: Detect gaps in parallel
        raw_gaps = await self._detect_gaps_parallel(root_query, shards, constants_context)
        logger.info(f"Step 2.3: Detected {len(raw_gaps)} raw gap candidates")

        if not raw_gaps:
            logger.info("No gaps detected, returning original coverage")
            return covered_chunks, {
                "gaps_found": 0,
                "gaps_filled": 0,
                "chunks_added": 0,
            }

        # Step 2.4a: Embed gap queries
        gap_embeddings = await self._embed_gap_queries(raw_gaps)

        # Step 2.4b: Cluster gap queries by similarity
        cluster_labels = self._cluster_gap_queries(gap_embeddings)
        num_clusters = len(set(cluster_labels))
        logger.info(
            f"Step 2.4: Clustered {len(raw_gaps)} gaps into {num_clusters} groups"
        )

        # Step 2.5: Unify gap clusters
        unified_gaps = await self._unify_gap_clusters(
            root_query, raw_gaps, cluster_labels
        )
        logger.info(f"Step 2.5: Unified to {len(unified_gaps)} gap queries")

        # Step 2.6: Select gaps by elbow detection
        selected_gaps = self._select_gaps_by_elbow(unified_gaps)
        logger.info(f"Step 2.6: Selected {len(selected_gaps)} gaps to fill")

        if not selected_gaps:
            logger.info("No gaps passed selection, returning original coverage")
            return covered_chunks, {
                "gaps_found": len(raw_gaps),
                "gaps_unified": len(unified_gaps),
                "gaps_filled": 0,
                "chunks_added": 0,
            }

        # Step 2.7: Fill gaps in parallel (INDEPENDENT - no shared mutable state)
        gap_results = await self._fill_gaps_parallel(
            root_query, selected_gaps, phase1_threshold, path_filter
        )

        # Step 2.8: Global deduplication (SYNC POINT)
        unified_gap_chunks = self._global_dedup(gap_results)
        total_before_dedup = sum(len(r) for r in gap_results)
        logger.info(
            f"Step 2.8: Global dedup: {total_before_dedup} → "
            f"{len(unified_gap_chunks)} unique chunks"
        )

        # Step 2.9: Merge coverage + gap chunks
        all_chunks = self._merge_coverage(covered_chunks, unified_gap_chunks)
        logger.info(
            f"Step 2.9: Final merge: {len(covered_chunks)} + "
            f"{len(unified_gap_chunks)} → {len(all_chunks)} total"
        )

        # Step 2.10: Import resolution (if enabled)
        import_chunks_added = 0
        if self._config.import_resolution_enabled and self._import_resolver:
            import_chunks = await resolve_and_fetch_imports(
                chunks=all_chunks,
                import_resolver=self._import_resolver,
                db_services=self._db_services,
                config=self._config,
                path_filter=path_filter,
                default_score=IMPORT_DEFAULT_SCORE,
            )
            if import_chunks:
                all_chunks = self._merge_coverage(all_chunks, import_chunks)
                import_chunks_added = len(import_chunks)
                logger.info(
                    f"Step 2.10: Import resolution: added {import_chunks_added} chunks "
                    f"from {len({c.get('file_path') for c in import_chunks})} import files"
                )

        # Extract gap queries for compound context in synthesis
        gap_queries = [gap.query for gap in selected_gaps]

        gap_stats = {
            "gaps_found": len(raw_gaps),
            "gaps_unified": len(unified_gaps),
            "gaps_selected": len(selected_gaps),
            "gaps_filled": len([r for r in gap_results if r]),
            "chunks_added": len(unified_gap_chunks),
            "import_chunks_added": import_chunks_added,
            "total_chunks": len(all_chunks),
            "gap_queries": gap_queries,  # For compound context in Phase 3
        }

        return all_chunks, gap_stats

    async def _cluster_chunks_kmeans(self, chunks: list[dict]) -> list[list[dict]]:
        """Step 2.1: Cluster chunks using k-means with token-based cluster count.

        Uses ceil(total_tokens / 50k) to determine cluster count.
        Skips clustering if total_tokens <= 50k.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of cluster groups (each group is a list of chunks)
        """
        if len(chunks) < 2:
            return [chunks]

        # Calculate total tokens to determine cluster count
        llm = self._llm_manager.get_utility_provider()
        total_tokens = sum(
            llm.estimate_tokens(get_chunk_text(chunk)) for chunk in chunks
        )

        # Skip clustering if content fits in single cluster budget
        if total_tokens <= GAP_CLUSTER_TOKEN_BUDGET:
            logger.debug(
                f"K-means: {len(chunks)} chunks ({total_tokens:,} tokens) "
                f"fits in single cluster, skipping"
            )
            return [chunks]

        # Generate embeddings for chunk content
        try:
            embedding_provider = self._embedding_manager.get_provider()
            chunk_contents = [get_chunk_text(chunk) for chunk in chunks]
            embeddings_list = await embedding_provider.embed_batch(chunk_contents)
            embeddings_array = np.array(embeddings_list)
        except Exception as e:
            logger.warning(f"Failed to generate embeddings for clustering: {e}")
            return [chunks]  # Fallback to single cluster

        # Calculate cluster count based on token budget
        num_clusters = math.ceil(total_tokens / GAP_CLUSTER_TOKEN_BUDGET)
        num_clusters = min(num_clusters, len(chunks))  # Can't exceed chunk count

        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=KMEANS_N_INIT)
        try:
            labels = kmeans.fit_predict(embeddings_array)
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}, using single cluster")
            return [chunks]

        # Group chunks by cluster label
        cluster_map: dict[int, list[dict]] = {}
        for chunk, label in zip(chunks, labels):
            cluster_id = int(label)
            cluster_map.setdefault(cluster_id, []).append(chunk)

        cluster_groups = list(cluster_map.values())
        logger.debug(
            f"K-means: {len(chunks)} chunks → {len(cluster_groups)} clusters "
            f"(target: {num_clusters})"
        )

        return cluster_groups

    def _shard_by_tokens(self, cluster_groups: list[list[dict]]) -> list[list[dict]]:
        """Step 2.2: Partition chunks into shards by token budget.

        Args:
            cluster_groups: List of cluster groups

        Returns:
            List of shards (each shard is a list of chunks within budget)
        """
        llm = self._llm_manager.get_utility_provider()
        shard_budget = self._config.shard_budget

        shards: list[list[dict]] = []
        current_shard: list[dict] = []
        current_tokens = 0

        for cluster in cluster_groups:
            for chunk in cluster:
                chunk_tokens = llm.estimate_tokens(get_chunk_text(chunk))

                # If adding this chunk exceeds budget, start new shard
                if current_tokens + chunk_tokens > shard_budget and current_shard:
                    shards.append(current_shard)
                    current_shard = []
                    current_tokens = 0

                current_shard.append(chunk)
                current_tokens += chunk_tokens

        # Add final shard
        if current_shard:
            shards.append(current_shard)

        if shards:
            logger.debug(
                f"Sharding: {len(cluster_groups)} clusters → {len(shards)} shards "
                f"(avg {sum(len(s) for s in shards) / len(shards):.1f} chunks/shard)"
            )
        else:
            logger.debug("Sharding: 0 clusters → 0 shards")

        return shards

    async def _detect_gaps_parallel(
        self,
        root_query: str,
        shards: list[list[dict]],
        constants_context: str = "",
    ) -> list[GapCandidate]:
        """Step 2.3: Detect gaps in parallel across shards.

        CRITICAL: Includes ROOT query in every LLM prompt to maintain focus.

        Args:
            root_query: Original research query
            shards: List of chunk shards
            constants_context: Constants ledger context for LLM prompts

        Returns:
            List of gap candidates from all shards
        """
        llm = self._llm_manager.get_utility_provider()
        # Get concurrency limit from utility provider
        max_concurrency = llm.get_synthesis_concurrency()
        semaphore = asyncio.Semaphore(max_concurrency)

        # JSON schema for gap detection
        schema = {
            "type": "object",
            "properties": {
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["query", "rationale", "confidence"],
                    },
                }
            },
            "required": ["gaps"],
        }

        # Create shared builder for code context (reused across shards)
        builder = ChunkContextBuilder(
            import_context_service=self._import_context_service,
            llm_manager=self._llm_manager,
        )
        shard_budget = self._config.shard_budget

        async def detect_gaps_in_shard(
            shard_idx: int, shard: list[dict]
        ) -> list[GapCandidate]:
            """Detect gaps in a single shard."""
            async with semaphore:
                # Build code context with imports using shared builder
                code_context = builder.build_code_context_with_imports(
                    shard, max_tokens=shard_budget
                )

                # Build constants section if available
                # Note: Gap detection uses inline instruction in prompt, not separate section
                constants_section = ""
                if constants_context:
                    constants_section = f"\n{constants_context}\n"

                # Gap detection prompt with ROOT QUERY
                prompt = f"""RESEARCH QUERY: {root_query}
{constants_section}
Given the research query above, identify semantic gaps in this code coverage.
Gaps: missing dependencies, incomplete flows, referenced-but-unfound components
that would help answer the RESEARCH QUERY.
For each gap, assess confidence (0.0-1.0) based on relevance to the query.

{CONSTANTS_INSTRUCTION_SHORT}

CODE COVERAGE:
{code_context}

Output JSON with gaps array."""

                try:
                    result = await llm.complete_structured(
                        prompt=prompt,
                        json_schema=schema,
                        max_completion_tokens=2048,
                    )

                    gaps = result.get("gaps", [])
                    gap_candidates = [
                        GapCandidate(
                            query=g["query"],
                            rationale=g["rationale"],
                            confidence=g["confidence"],
                            source_shard=shard_idx,
                        )
                        for g in gaps
                    ]

                    logger.debug(f"Shard {shard_idx}: Detected {len(gap_candidates)} gaps")
                    return gap_candidates

                except Exception as e:
                    logger.warning(f"Gap detection failed for shard {shard_idx}: {e}")
                    return []

        # Run gap detection in parallel
        tasks = [detect_gaps_in_shard(i, shard) for i, shard in enumerate(shards)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_gaps: list[GapCandidate] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Gap detection task failed: {result}")
                continue
            if isinstance(result, list):
                all_gaps.extend(result)

        return all_gaps

    async def _embed_gap_queries(self, gaps: list[GapCandidate]) -> np.ndarray:
        """Step 2.4a: Embed gap queries for clustering.

        Args:
            gaps: List of gap candidates

        Returns:
            Numpy array of embeddings
        """
        if not gaps:
            return np.array([])

        embedding_provider = self._embedding_manager.get_provider()
        queries = [gap.query for gap in gaps]

        embeddings = await embedding_provider.embed(queries)
        return np.array(embeddings)

    def _cluster_gap_queries(self, embeddings: np.ndarray) -> np.ndarray:
        """Step 2.4b: Cluster gap queries by cosine distance.

        Args:
            embeddings: Array of gap query embeddings

        Returns:
            Array of cluster labels
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))

        # AgglomerativeClustering with distance threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self._config.gap_similarity_threshold,
            metric="cosine",
            linkage="average",
        )

        labels = clusterer.fit_predict(embeddings)
        return labels  # type: ignore[no-any-return]

    async def _unify_gap_clusters(
        self, root_query: str, gaps: list[GapCandidate], labels: np.ndarray
    ) -> list[UnifiedGap]:
        """Step 2.5: Unify similar gaps using LLM.

        CRITICAL: Includes ROOT query in every unification prompt.

        Args:
            root_query: Original research query
            gaps: List of gap candidates
            labels: Cluster labels

        Returns:
            List of unified gaps
        """
        llm = self._llm_manager.get_utility_provider()

        # Group gaps by cluster label
        cluster_map: dict[int, list[GapCandidate]] = {}
        for gap, label in zip(gaps, labels):
            cluster_id = int(label)
            cluster_map.setdefault(cluster_id, []).append(gap)

        # Unification schema
        schema = {
            "type": "object",
            "properties": {
                "unified_query": {"type": "string"},
            },
            "required": ["unified_query"],
        }

        unified_gaps: list[UnifiedGap] = []

        for cluster_id, cluster_gaps in cluster_map.items():
            if len(cluster_gaps) == 1:
                # Single gap, no need to unify
                gap = cluster_gaps[0]
                # Apply same scoring formula: vote_count * avg_confidence * (1 + 0.3 * shard_bonus)
                vote_count = 1
                avg_confidence = gap.confidence
                shard_bonus = 1 / (1 + gap.source_shard)
                score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)
                unified_gaps.append(
                    UnifiedGap(
                        query=gap.query,
                        sources=cluster_gaps,
                        vote_count=vote_count,
                        avg_confidence=avg_confidence,
                        score=score,
                    )
                )
                continue

            # Multiple gaps - unify with LLM
            gap_list = "\n".join(
                f"- {g.query} (confidence: {g.confidence:.2f})" for g in cluster_gaps
            )

            prompt = f"""RESEARCH QUERY: {root_query}

Merge these similar gap queries into ONE refined query
that best addresses the research query above:

{gap_list}

Output a single unified query that captures the essential information need."""

            try:
                result = await llm.complete_structured(
                    prompt=prompt,
                    json_schema=schema,
                    max_completion_tokens=512,
                )

                unified_query = result.get("unified_query", cluster_gaps[0].query)
                vote_count = len(cluster_gaps)
                avg_confidence = sum(g.confidence for g in cluster_gaps) / vote_count

                # Score calculation with shard bonus
                min_shard_idx = min(g.source_shard for g in cluster_gaps)
                shard_bonus = 1 / (1 + min_shard_idx)
                score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)

                unified_gaps.append(
                    UnifiedGap(
                        query=unified_query,
                        sources=cluster_gaps,
                        vote_count=vote_count,
                        avg_confidence=avg_confidence,
                        score=score,
                    )
                )

                logger.debug(
                    f"Unified {vote_count} gaps (score: {score:.2f}): "
                    f"{unified_query[:60]}..."
                )

            except Exception as e:
                logger.warning(f"Gap unification failed for cluster {cluster_id}: {e}")
                # Fallback: use first gap with proper scoring formula
                gap = cluster_gaps[0]
                vote_count = len(cluster_gaps)
                avg_confidence = sum(g.confidence for g in cluster_gaps) / vote_count
                min_shard_idx = min(g.source_shard for g in cluster_gaps)
                shard_bonus = 1 / (1 + min_shard_idx)
                score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)
                unified_gaps.append(
                    UnifiedGap(
                        query=gap.query,
                        sources=cluster_gaps,
                        vote_count=vote_count,
                        avg_confidence=avg_confidence,
                        score=score,
                    )
                )

        return unified_gaps

    def _select_gaps_by_elbow(self, unified_gaps: list[UnifiedGap]) -> list[UnifiedGap]:
        """Step 2.6: Select gaps using elbow detection on scores.

        Uses kneedle algorithm to find natural break point in score curve.
        Fallback to 50% threshold heuristic if kneedle fails.

        Args:
            unified_gaps: List of unified gaps

        Returns:
            Selected gaps to fill
        """
        if not unified_gaps:
            return []

        # Sort by score descending
        sorted_gaps = sorted(unified_gaps, key=lambda g: g.score, reverse=True)

        # Apply min/max constraints
        min_gaps = self._config.min_gaps
        max_gaps = self._config.max_gaps

        # Special case: fewer gaps than minimum
        if len(sorted_gaps) <= min_gaps:
            return sorted_gaps

        # Special case: all gaps fit within max
        if len(sorted_gaps) <= max_gaps:
            # Try kneedle algorithm to find natural break
            elbow_idx = self._find_elbow_kneedle(sorted_gaps)
            if elbow_idx is not None and elbow_idx >= min_gaps:
                selected = sorted_gaps[:elbow_idx]
                # Apply 50% threshold as post-filter (but keep at least min_gaps)
                if sorted_gaps[0].score > 1e-9:
                    threshold = 0.5 * sorted_gaps[0].score
                    filtered = [g for g in selected if g.score >= threshold]
                    if len(filtered) >= min_gaps:
                        selected = filtered
                logger.debug(
                    f"Gap selection: {len(sorted_gaps)} candidates → "
                    f"{len(selected)} selected (kneedle elbow at {elbow_idx})"
                )
                return selected

            # Fallback to 50% threshold heuristic
            selected = sorted_gaps[:min_gaps]  # At least min_gaps

            # Guard: If top score is near-zero, no confidence to select beyond min_gaps
            if sorted_gaps[0].score < 1e-9:
                logger.debug(
                    f"Gap selection: {len(sorted_gaps)} candidates → "
                    f"{len(selected)} selected (near-zero top score)"
                )
                return selected

            for i in range(min_gaps, len(sorted_gaps)):
                # Stop if score drops below 50% of top score
                if sorted_gaps[i].score < 0.5 * sorted_gaps[0].score:
                    break
                selected.append(sorted_gaps[i])
            logger.debug(
                f"Gap selection: {len(sorted_gaps)} candidates → "
                f"{len(selected)} selected (fallback 50% heuristic)"
            )
            return selected

        # More gaps than max: use kneedle on first max_gaps
        candidate_gaps = sorted_gaps[:max_gaps]

        # Guard: If top score is near-zero, no confidence to select beyond min_gaps
        if sorted_gaps[0].score < 1e-9:
            selected = sorted_gaps[:min_gaps]
            logger.debug(
                f"Gap selection: {len(sorted_gaps)} candidates → "
                f"{len(selected)} selected (near-zero top score, exceeds max)"
            )
            return selected

        elbow_idx = self._find_elbow_kneedle(candidate_gaps)
        if elbow_idx is not None and elbow_idx >= min_gaps:
            selected = candidate_gaps[:elbow_idx]
            # Apply 50% threshold as post-filter (but keep at least min_gaps)
            threshold = 0.5 * sorted_gaps[0].score
            filtered = [g for g in selected if g.score >= threshold]
            if len(filtered) >= min_gaps:
                selected = filtered
            logger.debug(
                f"Gap selection: {len(sorted_gaps)} candidates → "
                f"{len(selected)} selected (kneedle elbow at {elbow_idx})"
            )
            return selected

        # Fallback: take max_gaps
        logger.debug(
            f"Gap selection: {len(sorted_gaps)} candidates → "
            f"{max_gaps} selected (fallback to max_gaps)"
        )
        return sorted_gaps[:max_gaps]

    def _find_elbow_kneedle(self, sorted_gaps: list[UnifiedGap]) -> int | None:
        """Find elbow point in score curve using simplified kneedle algorithm.

        Delegates to shared elbow_detection.find_elbow_kneedle() for DRY.

        Args:
            sorted_gaps: Gaps sorted by score descending

        Returns:
            Index of elbow point (1-based, returns 1 to select first gap only),
            or None if no clear elbow detected
        """
        # Extract scores and delegate to shared implementation
        scores = [gap.score for gap in sorted_gaps]
        elbow_idx = find_elbow_kneedle(scores)

        if elbow_idx is None:
            return None

        # Convert 0-based index to 1-based count (number of gaps to select)
        # We want to include the elbow point itself
        return elbow_idx + 1

    async def _fill_gaps_parallel(
        self,
        root_query: str,
        selected_gaps: list[UnifiedGap],
        phase1_threshold: float,
        path_filter: str | None,
    ) -> list[list[dict]]:
        """Step 2.7: Fill gaps in parallel using unified search.

        CRITICAL: Gap fills are INDEPENDENT - no shared mutable state.
        Each gap fill is a complete unified search with its own deduplication.

        Args:
            root_query: Original research query
            selected_gaps: Gaps to fill
            phase1_threshold: Quality threshold from Phase 1 (floor)
            path_filter: Path filter for searches

        Returns:
            List of gap result lists (one per gap)
        """

        async def fill_single_gap(gap: UnifiedGap) -> list[dict]:
            """Fill a single gap independently."""
            return await self._fill_single_gap(
                root_query, gap, phase1_threshold, path_filter
            )

        # Run all gap fills in parallel
        tasks = [fill_single_gap(gap) for gap in selected_gaps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        gap_results: list[list[dict]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Gap fill failed for '{selected_gaps[i].query[:60]}...': {result}"
                )
                gap_results.append([])
            elif isinstance(result, list):
                gap_results.append(result)
                logger.debug(
                    f"Gap fill complete: '{selected_gaps[i].query[:60]}...' → "
                    f"{len(result)} chunks"
                )

        return gap_results

    async def _fill_single_gap(
        self,
        root_query: str,
        gap: UnifiedGap,
        phase1_threshold: float,
        path_filter: str | None,
    ) -> list[dict]:
        """Step 2.7: Fill a single gap using unified search.

        This is INDEPENDENT - no shared mutable state between gap fills.

        Args:
            root_query: Original research query (for reranking)
            gap: Gap to fill
            phase1_threshold: Quality threshold floor
            path_filter: Path filter for search

        Returns:
            List of chunks for this gap (deduplicated, reranked, filtered)
        """
        # Create research context for unified search
        context = ResearchContext(root_query=root_query)

        # Run unified search for this gap with compound reranking
        # Per spec lines 327, 367-368: rerank against ROOT + gap_query
        chunks = await self._unified_search.unified_search(
            query=gap.query,
            context=context,
            rerank_queries=[root_query, gap.query],
            path_filter=path_filter,
        )

        # Apply window expansion if enabled
        if self._config.window_expansion_enabled:
            chunks = await self._unified_search.expand_chunk_windows(
                chunks,
                window_lines=self._config.window_expansion_lines
            )

        # Compute adaptive threshold for this gap
        if chunks:
            scores = [c.get("rerank_score", 0.0) for c in chunks]
            gap_threshold = compute_elbow_threshold(scores)
        else:
            gap_threshold = phase1_threshold

        # Apply threshold floor
        effective_threshold = max(phase1_threshold, gap_threshold)

        # Filter chunks by threshold
        filtered_chunks = [
            c for c in chunks if c.get("rerank_score", 0.0) >= effective_threshold
        ]

        logger.debug(
            f"Gap '{gap.query[:60]}...': {len(chunks)} → {len(filtered_chunks)} chunks "
            f"(threshold: {effective_threshold:.3f})"
        )

        return filtered_chunks

    def _global_dedup(self, gap_results: list[list[dict]]) -> list[dict]:
        """Step 2.8: Global deduplication across all gap results.

        SYNC POINT: This happens ONLY after all gap fills complete.
        Conflict resolution: keep chunk with highest rerank_score.

        Args:
            gap_results: List of gap result lists

        Returns:
            Deduplicated chunks
        """
        return deduplicate_chunks(gap_results, log_prefix="Gap dedup")

    def _merge_coverage(
        self, covered_chunks: list[dict], gap_chunks: list[dict]
    ) -> list[dict]:
        """Step 2.9: Merge coverage and gap chunks.

        Args:
            covered_chunks: Chunks from Phase 1
            gap_chunks: Chunks from gap filling

        Returns:
            Merged and deduplicated chunks
        """
        return merge_chunk_lists(
            covered_chunks, gap_chunks, log_prefix="Gap coverage merge"
        )
