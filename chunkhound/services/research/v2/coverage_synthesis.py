"""Coverage Synthesis Engine for Phase 3 of Coverage-First Research (v2).

This module implements the synthesis phase of the v2 coverage-first algorithm,
which combines chunks into a final answer using compound query context.

Key features:
- Compound reranking: Rerank files against ROOT + gap queries (prevents
  gap under-ranking)
- File reading: Read ALL prioritized files (no budget filtering)
- Boundary expansion: Expand chunks to complete functions/classes
- Recursive compression: Hierarchical map-reduce compression to fit output budget
- Final synthesis: Generate answer with ROOT + gap query context
- Citation management: Track and validate file references

Budget Philosophy:
    - target_tokens constrains OUTPUT size only, not input files
    - Compression handles any input size via recursive hierarchical map-reduce
    - All retrieved chunks are included in synthesis pipeline

Algorithm Flow:
    3.1 Compound Rerank: Rerank files against ROOT + gap queries
    3.2 File Reading: Read ALL files (compression handles size)
    3.3 Boundary Expansion: Expand to complete functions/classes
    3.4-3.5 Recursive Compression: Hierarchical map-reduce until budget met
    3.6 Final Synthesis: Generate answer with citations

Algorithm Foundation:
    The compression algorithm implements hierarchical map-reduce based on:
    - ToM (Tree-oriented MapReduce for Long-Context Reasoning), EMNLP 2025
      https://aclanthology.org/2025.emnlp-main.899.pdf
    - RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
      https://arxiv.org/abs/2401.18059

    Unlike flat iterative approaches, this uses recursive compression where
    each cluster is compressed to fit its budget BEFORE merging with siblings.
    This guarantees no single LLM call receives more than MAX_SINGLE_CALL_TOKENS.

Architecture:
    - Uses CitationManager for all citation handling
    - Uses FileReader for boundary expansion
    - Uses ClusteringService for compression clustering
    - Integrates with existing LLM and embedding providers
"""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.services.research.shared.constants_ledger import (
    CONSTANTS_INSTRUCTION_FULL,
)
from chunkhound.llm_manager import LLMManager
from chunkhound.services.prompts import (
    SYNTHESIS_SYSTEM_BUILDER,
    SYNTHESIS_USER,
)
from chunkhound.services.research.shared.chunk_dedup import merge_chunk_lists
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.models import (
    IMPORT_SYNTHESIS_SCORE,
    SINGLE_PASS_TIMEOUT_SECONDS,
)
from chunkhound.services.research.v2.compression_service import CompressionService


class CoverageSynthesisEngine:
    """Synthesis engine for Coverage-First Research (v2).

    Implements Phase 3 of the coverage-first research algorithm, combining
    chunks from Phase 2 into a final answer using compound query context
    (ROOT + gap queries).

    Key responsibilities:
        1. Compound reranking: Prevent gap-filling chunks from being under-ranked
        2. File reading: Read ALL prioritized files (no budget filtering)
        3. Boundary expansion: Complete functions/classes for better context
        4. Compression loop: Iterative map-reduce compression to fit budget
        5. Final synthesis: Generate answer with compound query context
        6. Citation management: Track and validate file references

    Budget Philosophy:
        - target_tokens constrains OUTPUT size only, not input filtering
        - Compression loop handles any input size via iterative map-reduce
        - All retrieved chunks are included in synthesis pipeline

    Compound Query Usage:
        - Reranking (3.1): Average scores across ROOT + gap queries
        - Compression (3.5): Include ROOT + gap queries in cluster prompts
        - Final synthesis (3.6): Include ROOT + gap queries in prompt
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: Any,
        db_services: DatabaseServices,
        config: Any,
        unified_search: Any | None = None,
        import_resolver: Any | None = None,
        import_context_service: ImportContextService | None = None,
        progress: Any | None = None,
    ):
        """Initialize coverage synthesis engine.

        Args:
            llm_manager: LLM manager for synthesis providers
            embedding_manager: Embedding manager for reranking
            db_services: Database services for file access
            config: ResearchConfig with synthesis parameters
            unified_search: Optional UnifiedSearch for window expansion
            import_resolver: Optional ImportResolverService for import resolution
            import_context_service: Optional ImportContextService for header injection
            progress: Optional progress callback for emitting events
        """
        self._llm_manager = llm_manager
        self._embedding_manager = embedding_manager
        self._db_services = db_services
        self._config = config
        self._unified_search = unified_search
        self._import_resolver = import_resolver
        self._import_context_service = import_context_service
        self._progress = progress

        # Initialize helpers
        self._citation_manager = CitationManager()
        self._file_reader = FileReader(db_services)
        self._compression_service = CompressionService(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            config=config,
            progress=progress,
        )

    async def synthesize(
        self,
        root_query: str,
        all_chunks: list[dict],
        gap_queries: list[str],
        target_tokens: int,
        file_imports: dict[str, list[str]] | None = None,
        constants_context: str = "",
    ) -> tuple[str, list[dict], dict]:
        """Synthesize final answer from coverage + gap chunks.

        Implements Phase 3 of the coverage-first research algorithm:
        1. Compound rerank files against ROOT + gap queries
        2. Read ALL prioritized files (no budget filtering)
        3. Expand boundaries to complete functions/classes
        4. Compress iteratively until budget met (REQUIRED)
        5. Generate final answer with compound query context

        The iterative map-reduce compression loop handles any input size.
        Budget (target_tokens) only constrains the final synthesis output,
        NOT the input files included.

        Args:
            root_query: Original research query
            all_chunks: Raw chunk dicts from Phase 2 (coverage + gap chunks)
            gap_queries: Gap queries that were filled (for compound context)
            target_tokens: Output token budget for final synthesis (not input filter)
            file_imports: Pre-extracted imports per file (avoids re-extraction
                during compression). Keys are file paths, values are import lists.
            constants_context: Constants ledger context for LLM prompts

        Returns:
            Tuple of (answer, citations, stats) where:
                - answer: Final synthesized response with sources footer
                - citations: List of file paths referenced
                - stats: Synthesis statistics (files read, tokens used, etc.)

        Raises:
            RuntimeError: If compression loop fails to converge within max iterations
        """
        # Use passed file_imports or empty dict
        _file_imports = file_imports or {}
        logger.info(
            f"Starting Phase 3 synthesis: {len(all_chunks)} chunks, "
            f"{len(gap_queries)} gap queries, target={target_tokens:,} tokens"
        )

        # Step 3.1: Compound rerank files
        file_priorities = await self._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        # Step 3.1.5: Import resolution (if enabled)
        import_chunks_added = 0
        if self._config.import_resolution_enabled and self._import_resolver:
            import_chunks = await resolve_and_fetch_imports(
                chunks=all_chunks,
                import_resolver=self._import_resolver,
                db_services=self._db_services,
                config=self._config,
                path_filter=None,  # Phase 3 doesn't use path filter
                default_score=IMPORT_SYNTHESIS_SCORE,  # Phase 3 uses lower score for late discovery
            )
            if import_chunks:
                # Merge import chunks with all_chunks for budget allocation
                all_chunks = self._merge_chunks(all_chunks, import_chunks)
                import_chunks_added = len(import_chunks)
                logger.info(
                    f"Step 3.1.5: Import resolution: added {import_chunks_added} chunks "
                    f"from {len({c.get('file_path') for c in import_chunks})} import files"
                )

                # Rerank the new chunks
                file_priorities = await self._compound_rerank_files(
                    root_query, all_chunks, gap_queries
                )

        # Step 3.1.6: Window expansion (if enabled)
        window_chunks_added = 0
        if self._config.window_expansion_enabled and self._unified_search:
            original_count = len(all_chunks)
            all_chunks = await self._unified_search.expand_chunk_windows(
                all_chunks,
                window_lines=self._config.window_expansion_lines,
            )
            window_chunks_added = len(all_chunks) - original_count
            if window_chunks_added > 0:
                logger.info(
                    f"Step 3.1.6: Window expansion added {window_chunks_added} "
                    f"neighboring chunks ({original_count} -> {len(all_chunks)})"
                )

        # Step 3.2: Read all prioritized files (no budget filtering)
        all_files, read_stats = await self._read_prioritized_files(
            all_chunks, file_priorities
        )

        # Step 3.3: Expand boundaries
        expanded_files = await self._expand_boundaries(all_files)

        # Step 3.4-3.5: Recursive compression (hierarchical map-reduce)
        compressed_content = await self._compression_service.compress_to_budget(
            root_query, gap_queries, expanded_files, target_tokens, _file_imports
        )

        # Step 3.6: Final synthesis
        answer = await self._final_synthesis(
            root_query,
            gap_queries,
            compressed_content,
            all_chunks,
            expanded_files,
            _file_imports,
            constants_context,
        )

        # Build citations
        citations = self._build_citations(all_chunks, expanded_files)

        # Build stats
        stats = {
            **read_stats,
            "target_tokens": target_tokens,
            "final_tokens": self._llm_manager.get_utility_provider().estimate_tokens(
                answer
            ),
            "gap_queries": gap_queries,
            "num_gap_queries": len(gap_queries),
            "import_chunks_added": import_chunks_added,
            "window_chunks_added": window_chunks_added,
        }

        logger.info(
            f"Phase 3 synthesis complete: {stats['final_tokens']:,} tokens generated"
        )

        return answer, citations, stats

    async def _compound_rerank_files(
        self,
        root_query: str,
        all_chunks: list[dict],
        gap_queries: list[str],
    ) -> dict[str, float]:
        """Rerank files against ROOT + gap queries (step 3.1).

        Prevents gap-filling chunks from being under-ranked by using compound
        query context. Averages rerank scores across all queries.

        Args:
            root_query: Original research query
            all_chunks: All chunks from Phase 2
            gap_queries: Gap queries that were filled

        Returns:
            Dictionary mapping file_path -> average rerank score
        """
        logger.info(
            f"Compound reranking files against {1 + len(gap_queries)} queries "
            f"(root + {len(gap_queries)} gaps)"
        )

        # Build compound queries
        compound_queries = [root_query] + gap_queries

        # Build file representatives (top 5 chunks per file, max 2000 tokens)
        llm = self._llm_manager.get_utility_provider()
        file_to_chunks: dict[str, list[dict]] = {}

        for chunk in all_chunks:
            file_path = chunk.get("file_path", "")
            if file_path:
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)

        # Create representatives
        file_paths = []
        file_documents = []

        for file_path, file_chunks in file_to_chunks.items():
            # Sort by score and take top chunks
            sorted_chunks = sorted(
                file_chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True
            )
            top_chunks = sorted_chunks[: self._config.max_chunks_per_file_repr]

            # Build representative document
            repr_parts = []
            for chunk in top_chunks:
                start_line = chunk.get("start_line", 1)
                end_line = chunk.get("end_line", 1)
                content = chunk.get("content", "")
                repr_parts.append(f"Lines {start_line}-{end_line}:\n{content}")

            document = f"{file_path}\n\n" + "\n\n".join(repr_parts)

            # Truncate to token limit
            if llm.estimate_tokens(document) > self._config.max_tokens_per_file_repr:
                chars_to_include = self._config.max_tokens_per_file_repr * 4
                document = document[:chars_to_include]

            file_paths.append(file_path)
            file_documents.append(document)

        # Rerank against each compound query
        embedding_provider = self._embedding_manager.get_provider()
        file_scores: dict[str, list[float]] = {fp: [] for fp in file_paths}

        for query in compound_queries:
            logger.debug(
                f"Reranking {len(file_documents)} files against: {query[:50]}..."
            )

            try:
                rerank_results = await embedding_provider.rerank(
                    query=query, documents=file_documents, top_k=None
                )

                # Store scores
                for result in rerank_results:
                    file_path = file_paths[result.index]
                    file_scores[file_path].append(result.score)

            except Exception as e:
                logger.warning(
                    f"Reranking failed for query '{query[:30]}...': {e}, "
                    f"using fallback scores"
                )
                # Fallback: Use uniform scores
                for file_path in file_paths:
                    file_scores[file_path].append(0.5)

        # Average scores across all queries
        file_priorities: dict[str, float] = {}
        for file_path, scores in file_scores.items():
            if scores:
                file_priorities[file_path] = sum(scores) / len(scores)
            else:
                # Fallback for files with no scores
                file_priorities[file_path] = 0.0

        logger.info(f"Compound reranking complete: {len(file_priorities)} files ranked")

        return file_priorities

    async def _read_prioritized_files(
        self,
        all_chunks: list[dict],
        file_priorities: dict[str, float],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Read all files sorted by compound rerank priority (step 3.2).

        Reads full file content for ALL files in the priority map.
        No budget filtering - compression loop handles size reduction.

        Args:
            all_chunks: All chunks from Phase 2
            file_priorities: File path -> rerank score mapping

        Returns:
            Tuple of (all_files, stats) where:
                - all_files: Dict mapping file_path -> content
                - stats: File reading statistics
        """
        logger.info(f"Reading {len(file_priorities)} prioritized files")

        llm = self._llm_manager.get_utility_provider()

        # Sort files by priority
        sorted_files = sorted(file_priorities.items(), key=lambda x: x[1], reverse=True)

        # Build file-to-chunks mapping
        file_to_chunks: dict[str, list[dict]] = {}
        for chunk in all_chunks:
            file_path = chunk.get("file_path", "")
            if file_path:
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)

        # Read ALL files (no budget filtering - compression loop handles size)
        all_files: dict[str, str] = {}
        total_tokens = 0
        files_read = 0
        files_failed = 0

        # Get base directory for file reading
        base_dir = self._db_services.provider.get_base_directory()  # type: ignore[attr-defined]

        for file_path, priority in sorted_files:
            if file_path not in file_to_chunks:
                continue

            try:
                # Read full file
                if Path(file_path).is_absolute():
                    path = Path(file_path)
                else:
                    path = base_dir / file_path

                if not path.exists():
                    logger.warning(f"File not found: {file_path}")
                    files_failed += 1
                    continue

                content = path.read_text(encoding="utf-8", errors="ignore")
                content_tokens = llm.estimate_tokens(content)

                # Include ALL files - no budget filtering
                all_files[file_path] = content
                total_tokens += content_tokens
                files_read += 1

            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                files_failed += 1
                continue

        read_stats = {
            "total_tokens": total_tokens,
            "files_read": files_read,
            "files_failed": files_failed,
            "total_files": len(sorted_files),
        }

        logger.info(
            f"File reading complete: {files_read} files, {total_tokens:,} tokens total"
        )

        return all_files, read_stats

    async def _expand_boundaries(
        self,
        budgeted_files: dict[str, str],
    ) -> dict[str, str]:
        """Expand chunks to complete functions/classes (step 3.3).

        Uses FileReader.expand_to_natural_boundaries() for smart expansion.
        Max expansion is configurable (default: 300 lines).

        For partial files (snippet-based content with "..." separator), expands
        incomplete functions/classes at boundaries by reading additional lines
        from disk. Full files are returned unchanged.

        Args:
            budgeted_files: Dict mapping file_path -> content (from budget allocation)

        Returns:
            Dictionary mapping file_path -> expanded content
        """
        logger.info(
            f"Expanding boundaries for {len(budgeted_files)} files "
            f"(max {self._config.max_boundary_expansion_lines} lines)"
        )

        expanded_files: dict[str, str] = {}
        stats = {
            "full_files": 0,
            "partial_files": 0,
            "files_expanded": 0,
            "total_lines_added": 0,
        }

        # Get base directory for path resolution
        base_dir = self._db_services.provider.get_base_directory()  # type: ignore[attr-defined]

        for file_path, content in budgeted_files.items():
            # Check if this is a full file or partial (snippets)
            is_full = self._file_reader.is_file_fully_read(content)

            if is_full:
                # Full file - no expansion needed
                expanded_files[file_path] = content
                stats["full_files"] += 1
                continue

            stats["partial_files"] += 1

            # Partial file - expand boundaries for incomplete functions/classes
            try:
                from pathlib import Path

                if Path(file_path).is_absolute():
                    path = Path(file_path)
                else:
                    path = base_dir / file_path

                if not path.exists():
                    logger.warning(
                        f"File not found for boundary expansion: {file_path}"
                    )
                    expanded_files[file_path] = content
                    continue

                # Read full file
                full_content = path.read_text(encoding="utf-8", errors="ignore")
                lines = full_content.split("\n")

                # Parse snippet sections to find boundaries
                sections = content.split("\n\n...\n\n")
                expanded_sections = []

                for section in sections:
                    # Extract line range from section header (format: "# Lines X-Y")
                    if section.startswith("# Lines "):
                        header_line = section.split("\n", 1)[0]
                        try:
                            # Parse "# Lines X-Y"
                            range_str = header_line.split("Lines ", 1)[1]
                            start_line, end_line = map(int, range_str.split("-"))

                            # Expand to natural boundaries
                            expanded_start, expanded_end = (
                                self._file_reader.expand_to_natural_boundaries(
                                    lines,
                                    start_line,
                                    end_line,
                                    {},  # No chunk metadata available
                                    file_path,
                                )
                            )

                            # Cap expansion to configured limit
                            max_lines = self._config.max_boundary_expansion_lines
                            if expanded_end - expanded_start > max_lines:
                                # Keep original center, expand within limits
                                mid = (start_line + end_line) // 2
                                expanded_start = max(1, mid - max_lines // 2)
                                expanded_end = min(
                                    len(lines), expanded_start + max_lines
                                )

                            # Extract expanded content
                            start_idx = max(0, expanded_start - 1)
                            end_idx = min(len(lines), expanded_end)
                            expanded_content = "\n".join(lines[start_idx:end_idx])

                            # Track expansion stats
                            original_lines = end_line - start_line + 1
                            expanded_lines = end_idx - start_idx
                            if expanded_lines > original_lines:
                                stats["files_expanded"] += 1
                                lines_added = expanded_lines - original_lines
                                stats["total_lines_added"] += lines_added

                            # Use header with expanded range
                            header = f"# Lines {expanded_start}-{expanded_end}"
                            expanded_sections.append(f"{header}\n{expanded_content}")

                        except (ValueError, IndexError) as e:
                            logger.debug(
                                f"Could not parse line range from section header: {e}"
                            )
                            expanded_sections.append(section)
                    else:
                        # Section without header, keep as-is
                        expanded_sections.append(section)

                # Combine expanded sections
                expanded_content = "\n\n...\n\n".join(expanded_sections)
                expanded_files[file_path] = expanded_content

            except Exception as e:
                logger.warning(f"Failed to expand boundaries for {file_path}: {e}")
                expanded_files[file_path] = content
                continue

        logger.info(
            f"Boundary expansion complete: {stats['full_files']} full files, "
            f"{stats['partial_files']} partial files, "
            f"{stats['files_expanded']} expanded "
            f"(+{stats['total_lines_added']} lines)"
        )

        return expanded_files

    async def _final_synthesis(
        self,
        root_query: str,
        gap_queries: list[str],
        compressed_content: dict[str, str],
        all_chunks: list[dict],
        original_files: dict[str, str],
        file_imports: dict[str, list[str]],
        constants_context: str = "",
    ) -> str:
        """Generate final answer with compound query context (step 3.6).

        Includes ROOT + gap queries in prompt to ensure comprehensive coverage.

        Args:
            root_query: Original research query
            gap_queries: Gap queries for compound context
            compressed_content: Compressed content from compression loop
            all_chunks: Original chunks for citation building
            original_files: Original files for citation building
            file_imports: Pre-extracted imports per file (lookup, no re-parsing)
            constants_context: Constants ledger context for LLM prompts

        Returns:
            Final synthesized answer with sources footer
        """
        logger.info("Generating final synthesis with compound query context")

        llm = self._llm_manager.get_synthesis_provider()

        # Build compound query context
        compound_context = f"PRIMARY QUERY: {root_query}"
        if gap_queries:
            compound_context += "\n\nRELATED GAPS IDENTIFIED:\n"
            for gap in gap_queries:
                compound_context += f"- {gap}\n"
            compound_context += (
                "\nAddress the primary query while incorporating "
                "insights from gap areas."
            )
        if constants_context:
            compound_context += f"\n\n{constants_context}\n\n{CONSTANTS_INSTRUCTION_FULL}"

        # Filter chunks to match files
        filtered_chunks = self._citation_manager.filter_chunks_to_files(
            all_chunks, original_files
        )

        # Build file reference map
        file_reference_map = self._citation_manager.build_file_reference_map(
            filtered_chunks, original_files
        )
        reference_table = self._citation_manager.format_reference_table(
            file_reference_map
        )

        # Inject imports for uncompressed files (using pre-extracted file_imports)
        # Cluster summaries (e.g., "cluster_0_0") won't be in file_imports, naturally skipped
        enriched_content = dict(compressed_content)
        for key, content in compressed_content.items():
            imports = file_imports.get(key)  # Simple lookup, no re-parsing
            if imports:
                imports_header = "# Imports:\n" + "\n".join(imports) + "\n\n"
                enriched_content[key] = imports_header + content

        # Build code context from compressed content
        code_sections = []
        for key, content in enriched_content.items():
            code_sections.append(f"### {key}\n{'=' * 80}\n{content}\n{'=' * 80}")

        code_context = "\n\n".join(code_sections)

        # Build synthesis prompt
        output_guidance = (
            f"**Target Output:** Provide a thorough and detailed analysis of approximately "
            f"{self._config.target_tokens:,} tokens (includes reasoning). Focus on all relevant "
            f"architectural layers, patterns, and implementation details with technical accuracy."
        )

        system = SYNTHESIS_SYSTEM_BUILDER(output_guidance)

        prompt = SYNTHESIS_USER.format(
            root_query=compound_context,  # Use compound context instead of root query
            reference_table=reference_table,
            code_context=code_context,
        )

        logger.info(
            f"Calling LLM for final synthesis "
            f"(max_completion_tokens={self._config.target_tokens:,})"
        )

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=self._config.target_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )

        answer = response.content

        # Validate minimum length
        min_synthesis_length = 100
        answer_length = len(answer.strip()) if answer else 0

        if answer_length < min_synthesis_length:
            logger.error(
                f"Synthesis returned suspiciously short answer: {answer_length} chars"
            )
            raise RuntimeError(
                f"LLM synthesis failed: generated only {answer_length} "
                f"characters (minimum: {min_synthesis_length}). "
                f"finish_reason={response.finish_reason}."
            )

        # Append sources footer
        try:
            footer = self._citation_manager.build_sources_footer(
                filtered_chunks, original_files, file_reference_map
            )
            if footer:
                answer = f"{answer}\n\n{footer}"
        except Exception as e:
            logger.warning(
                f"Failed to generate sources footer: {e}. Continuing without footer."
            )

        final_tokens = llm.estimate_tokens(answer)
        logger.info(f"Final synthesis complete: {final_tokens:,} tokens generated")

        return answer

    def _build_citations(
        self,
        all_chunks: list[dict],
        files: dict[str, str],
    ) -> list[dict]:
        """Build citation list from chunks and files.

        Args:
            all_chunks: All chunks used in synthesis
            files: Files used in synthesis

        Returns:
            List of citation dicts with file_path, start_line, end_line
        """
        # Filter chunks to match files
        filtered_chunks = self._citation_manager.filter_chunks_to_files(
            all_chunks, files
        )

        # Build citations
        citations = []
        for chunk in filtered_chunks:
            citations.append(
                {
                    "file_path": chunk.get("file_path"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                }
            )

        return citations

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        llm = self._llm_manager.get_utility_provider()
        return llm.estimate_tokens(text)

    def _merge_chunks(
        self,
        chunks1: list[dict],
        chunks2: list[dict],
    ) -> list[dict]:
        """Merge two chunk lists, deduplicating by chunk_id.

        Args:
            chunks1: First list of chunks
            chunks2: Second list of chunks

        Returns:
            Merged and deduplicated chunks (keeping highest rerank_score)
        """
        return merge_chunk_lists(chunks1, chunks2, log_prefix="Synthesis merge")
