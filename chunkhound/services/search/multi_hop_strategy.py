"""Multi-hop search strategy with dynamic expansion and reranking.

This strategy performs iterative search with relevance-based expansion:
1. Initial search: Get top candidates from HNSW and rerank
2. Dynamic expansion: Find similar chunks to top results
3. Iterative reranking: Re-score all candidates after each expansion
4. Smart termination: Stop when relevance stops improving

Termination conditions (5 total):
1. Time limit: 5 seconds maximum
2. Result limit: 500 chunks maximum
3. Candidate quality: Need 5+ high-scoring candidates for expansion
4. Score degradation: Stop if tracked chunk scores drop by >= 0.15
5. Minimum relevance: Stop if top-5 minimum score < 0.3

Best for:
- Complex queries requiring context exploration
- Providers with reranking capabilities (Cohere, Voyage, etc.)
- Cases where initial HNSW results miss relevant context
"""

import time
from typing import Any

from loguru import logger

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider


class MultiHopStrategy:
    """Dynamic multi-hop semantic search with relevance-based termination."""

    def __init__(
        self,
        database_provider: DatabaseProvider,
        embedding_provider: EmbeddingProvider,
        single_hop_search_fn,
    ):
        """Initialize multi-hop search strategy.

        Args:
            database_provider: Database provider for vector search and expansion
            embedding_provider: Embedding provider with reranking support
            single_hop_search_fn: Function to perform initial single-hop search
        """
        self._db = database_provider
        self._embedding_provider = embedding_provider
        self._single_hop_search = single_hop_search_fn

    async def search(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform dynamic multi-hop semantic search with reranking.

        Args:
            query: Natural language search query
            page_size: Number of results per page
            offset: Starting position for pagination
            threshold: Optional similarity threshold to filter results
            provider: Embedding provider name
            model: Embedding model name
            path_filter: Optional relative path to limit search scope

        Returns:
            Tuple of (results, pagination_metadata)
        """
        start_time = time.perf_counter()

        # Step 1: Initial search + rerank
        initial_limit = min(page_size * 3, 100)  # Cap at 100 for performance
        initial_results, _ = await self._single_hop_search(
            query=query,
            page_size=initial_limit,
            offset=0,
            threshold=0.0,
            provider=provider,
            model=model,
            path_filter=path_filter,
        )

        if len(initial_results) <= 5:
            # Not enough results for expansion, fall back to standard search
            logger.debug(
                "Not enough results for dynamic expansion, using standard search"
            )
            return await self._single_hop_search(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=threshold,
                provider=provider,
                model=model,
                path_filter=path_filter,
            )

        # Rerank initial results
        try:
            assert hasattr(self._embedding_provider, "rerank")

            # Initialize all results with similarity scores as baseline
            for result in initial_results:
                if "score" not in result:
                    result["score"] = result.get("similarity", 0.0)

            documents = [result["content"] for result in initial_results]
            rerank_results = await self._embedding_provider.rerank(
                query=query,
                documents=documents,
                top_k=len(documents),
            )

            # Apply reranking scores (rerank_result.index maps to documents array position)
            for rerank_result in rerank_results:
                if 0 <= rerank_result.index < len(initial_results):
                    initial_results[rerank_result.index]["score"] = rerank_result.score

            # Log reranking effectiveness
            reranked_count = len(rerank_results)
            logger.debug(
                f"Initial reranking: {reranked_count}/{len(initial_results)} results reranked"
            )

            # Sort by rerank score (highest first)
            initial_results = sorted(
                initial_results, key=lambda x: x.get("score", 0.0), reverse=True
            )
        except Exception as e:
            logger.warning(f"Initial reranking failed: {e}")
            # Ensure all results still have scores using similarity as fallback
            for result in initial_results:
                if "score" not in result:
                    result["score"] = result.get("similarity", 0.0)

        # Step 2: Dynamic expansion loop
        all_results = list(initial_results)
        seen_chunk_ids = {result["chunk_id"] for result in initial_results}
        # Track specific chunks and their scores (not positions)
        top_chunk_scores = {}
        for result in initial_results[:5]:
            top_chunk_scores[result["chunk_id"]] = result.get("score", 0.0)

        expansion_round = 0

        while True:
            # Check termination conditions
            if time.perf_counter() - start_time >= 5.0:
                logger.debug(
                    "Dynamic expansion terminated: 5 second time limit reached"
                )
                break
            if len(all_results) >= 500:
                logger.debug("Dynamic expansion terminated: 500 result limit reached")
                break

            # Get top 5 candidates for expansion
            top_candidates = [r for r in all_results if r.get("score", 0.0) > 0.0][:5]
            if len(top_candidates) < 5:
                logger.debug(
                    "Dynamic expansion terminated: insufficient high-scoring candidates"
                )
                break

            # Expand using find_similar_chunks for each top candidate
            new_candidates = []
            for candidate in top_candidates:
                try:
                    # logger.debug(f"Expanding chunk_id={candidate['chunk_id']} using provider='{provider}', model='{model}'")
                    neighbors = self._db.find_similar_chunks(
                        chunk_id=candidate["chunk_id"],
                        provider=provider,
                        model=model,
                        limit=20,  # Get more neighbors per round
                        threshold=None,
                        path_filter=path_filter,
                    )

                    # Filter out already seen chunks
                    for neighbor in neighbors:
                        if neighbor["chunk_id"] not in seen_chunk_ids:
                            new_candidates.append(neighbor)
                            seen_chunk_ids.add(neighbor["chunk_id"])

                    # logger.debug(f"Found {len(neighbors)} neighbors for chunk_id={candidate['chunk_id']}, "
                    #            f"{len([n for n in neighbors if n['chunk_id'] not in seen_chunk_ids])} new")

                except Exception as e:
                    logger.warning(
                        f"Failed to expand chunk {candidate['chunk_id']}: {e}"
                    )
                    # Continue with other candidates even if one fails

            if not new_candidates:
                logger.debug("Dynamic expansion terminated: no new candidates found")
                break

            # Add new candidates and rerank all results
            all_results.extend(new_candidates)

            try:
                # Initialize all results with scores (similarity fallback for new candidates)
                for result in all_results:
                    if "score" not in result:
                        result["score"] = result.get("similarity", 0.0)

                documents = [result["content"] for result in all_results]
                # Type narrowing: we know provider has rerank if we're in multi-hop
                assert hasattr(self._embedding_provider, "rerank")
                rerank_results = await self._embedding_provider.rerank(
                    query=query,
                    documents=documents,
                    top_k=len(documents),
                )

                # Apply reranking scores (rerank_result.index maps to documents array position)
                for rerank_result in rerank_results:
                    if 0 <= rerank_result.index < len(all_results):
                        all_results[rerank_result.index]["score"] = rerank_result.score

                # Log reranking effectiveness
                reranked_count = len(rerank_results)
                logger.debug(
                    f"Expansion reranking: {reranked_count}/{len(all_results)} results reranked"
                )

                # Sort by rerank score
                all_results = sorted(
                    all_results, key=lambda x: x.get("score", 0.0), reverse=True
                )

            except Exception as e:
                logger.warning(
                    f"Reranking failed in expansion round {expansion_round}: {e}"
                )
                # Scores already initialized, just sort and continue
                all_results = sorted(
                    all_results, key=lambda x: x.get("score", 0.0), reverse=True
                )
                break

            # Check score derivative for termination (track specific chunks, not positions)
            current_top_scores = [
                result.get("score", 0.0) for result in all_results[:5]
            ]

            # Check if any of the originally top chunks have degraded significantly
            score_drops = []
            if top_chunk_scores:  # Only check after first iteration
                for chunk_id, prev_score in top_chunk_scores.items():
                    # Find this chunk's current score
                    current_score = next(
                        (
                            r.get("score", 0.0)
                            for r in all_results
                            if r["chunk_id"] == chunk_id
                        ),
                        0.0,  # If not in results anymore, score is 0
                    )
                    if current_score < prev_score:
                        score_drops.append(prev_score - current_score)

            # Update tracked chunks to current top 5
            top_chunk_scores.clear()
            for result in all_results[:5]:
                top_chunk_scores[result["chunk_id"]] = result.get("score", 0.0)

            # Check termination conditions
            if score_drops and max(score_drops) >= 0.15:
                logger.debug(
                    f"Dynamic expansion terminated: tracked chunk score drop "
                    f"{max(score_drops):.3f} >= 0.15"
                )
                break

            if min(current_top_scores) < 0.3:
                logger.debug(
                    f"Dynamic expansion terminated: minimum score "
                    f"{min(current_top_scores):.3f} < 0.3"
                )
                break
            expansion_round += 1

            logger.debug(
                f"Expansion round {expansion_round}: {len(all_results)} total results"
            )

        # Step 3: Final filtering and pagination
        # In multi-hop search, threshold applies to rerank scores (not similarity scores)
        # since rerank scores are the final relevance metric after expansion
        if threshold is not None:
            # Use 0.0 default so unscored results are treated as low relevance, not perfect matches
            all_results = [r for r in all_results if r.get("score", 0.0) >= threshold]
            logger.debug(
                f"Applied rerank score threshold {threshold}, {len(all_results)} results remain"
            )

        # Apply pagination
        total_results = len(all_results)
        paginated_results = all_results[offset : offset + page_size]

        pagination = {
            "offset": offset,
            "page_size": page_size,
            "has_more": offset + page_size < total_results,
            "next_offset": offset + page_size
            if offset + page_size < total_results
            else None,
            "total": total_results,
        }

        elapsed_time = time.perf_counter() - start_time
        logger.info(
            f"Dynamic expansion search completed in {elapsed_time:.2f}s: "
            f"{len(paginated_results)} results returned "
            f"({total_results} total candidates, "
            f"{expansion_round} expansion rounds)"
        )
        return paginated_results, pagination
