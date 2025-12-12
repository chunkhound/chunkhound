from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services

from operations.deep_doc.deep_doc import _compute_db_scope_stats, _resolve_scope


@dataclass
class CoverageStats:
    # From document Sources footers (approximate; may double-count across sections)
    doc_sources_sections: int
    doc_sources_files_total: int
    doc_sources_chunks_total: int

    # From database (exact for given scope)
    scope_total_files_indexed: int
    scope_total_chunks_indexed: int

    # Approximate lower-bound coverage ratios based on summed Sources counts
    approx_file_coverage_percent: float | None
    approx_chunk_coverage_percent: float | None


def _parse_sources_from_doc(doc_text: str) -> tuple[int, int, int]:
    """Parse all '## Sources' sections in the doc and sum file/chunk counts.

    This relies on the standard footer format emitted by CitationManager:

        ---

        ## Sources

        **Files**: N | **Chunks**: M
        ...

    We intentionally do not try to deduplicate files across sections, since the
    agentic pipeline may embed multiple independent footer blocks from distinct
    code_research calls. The result is an upper bound on referenced files/
    chunks; coverage calculations treat it as a best-effort proxy.
    """
    lines = doc_text.splitlines()
    sections = 0
    total_files = 0
    total_chunks = 0

    in_sources = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Sources"):
            in_sources = True
            sections += 1
            continue

        if not in_sources:
            continue

        # Stop the current section when we hit another top-level heading.
        if stripped.startswith("## ") and not stripped.startswith("## Sources"):
            in_sources = False
            continue

        # Look for the summary line: **Files**: N | **Chunks**: M
        if "**Files**" in stripped and "**Chunks**" in stripped:
            match = re.search(
                r"\*\*Files\*\*:\s*(\d+)\s*\|\s*\*\*Chunks\*\*:\s*(\d+)", stripped
            )
            if match:
                files = int(match.group(1))
                chunks = int(match.group(2))
                total_files += files
                total_chunks += chunks

    return sections, total_files, total_chunks


def compute_coverage(
    *,
    project_root: Path,
    scope: str,
    doc_path: Path,
) -> CoverageStats:
    """Compute best-effort coverage metadata for an agentic Agent Doc."""
    project_root = project_root.resolve()
    doc_path = doc_path.resolve()

    if not doc_path.exists():
        raise SystemExit(f"Agent doc not found at {doc_path}")

    # Resolve scope folder and scope label to match deep_doc semantics.
    scope_path, scope_label = _resolve_scope(project_root, scope)

    # Load document text and parse Sources footers.
    doc_text = doc_path.read_text(encoding="utf-8")
    sections, doc_files_total, doc_chunks_total = _parse_sources_from_doc(doc_text)

    # Wire up ChunkHound config and DB services (no embeddings required).
    config_args = argparse.Namespace(path=str(project_root), config=None)
    config = Config(args=config_args)

    db_path = config.database.path
    if not db_path or not db_path.exists():
        raise SystemExit(
            f"Database not found at {db_path}. Run 'chunkhound index .' in the project root first."
        )

    services = create_services(db_path=db_path, config=config, embedding_manager=None)

    # Compute exact scope totals from the database.
    scope_total_files, scope_total_chunks, _ = _compute_db_scope_stats(
        services, scope_label
    )

    # Approximate coverage ratios: summed Sources counts vs scope totals.
    file_cov = (
        (min(doc_files_total, scope_total_files) / scope_total_files * 100.0)
        if scope_total_files > 0 and doc_files_total > 0
        else None
    )
    chunk_cov = (
        (min(doc_chunks_total, scope_total_chunks) / scope_total_chunks * 100.0)
        if scope_total_chunks > 0 and doc_chunks_total > 0
        else None
    )

    return CoverageStats(
        doc_sources_sections=sections,
        doc_sources_files_total=doc_files_total,
        doc_sources_chunks_total=doc_chunks_total,
        scope_total_files_indexed=scope_total_files,
        scope_total_chunks_indexed=scope_total_chunks,
        approx_file_coverage_percent=file_cov,
        approx_chunk_coverage_percent=chunk_cov,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute best-effort coverage metadata for an agentic Agent Doc "
            "against the ChunkHound database."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Project root path (where .chunkhound.json and DB live). Defaults to current directory.",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        required=True,
        help="Path to the Agent Doc markdown file to analyze.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="chunkhound",
        help=(
            "Relative folder within the project that the Agent Doc describes "
            "(e.g. 'chunkhound', '.' or '/' for entire workspace). "
            "Defaults to 'chunkhound'."
        ),
    )

    args = parser.parse_args()

    stats = compute_coverage(
        project_root=args.path,
        scope=args.scope,
        doc_path=args.doc,
    )

    # Human-readable summary; structured enough to be scraped if needed.
    print(f"Agent Doc: {args.doc}")
    print(f"Project root: {args.path.resolve()}")
    print(f"Scope: {args.scope}")
    print()
    print("Document Sources (approximate, summed across sections):")
    print(f"  sections_with_sources: {stats.doc_sources_sections}")
    print(f"  doc_sources_files_total: {stats.doc_sources_files_total}")
    print(f"  doc_sources_chunks_total: {stats.doc_sources_chunks_total}")
    print()
    print("Database scope totals:")
    print(f"  scope_total_files_indexed: {stats.scope_total_files_indexed}")
    print(f"  scope_total_chunks_indexed: {stats.scope_total_chunks_indexed}")
    print()
    print("Approximate lower-bound coverage:")
    if stats.approx_file_coverage_percent is not None:
        print(
            f"  file_coverage_percent: {stats.approx_file_coverage_percent:.2f}% "
            "(min(doc_files_total, scope_total_files) / scope_total_files)"
        )
    else:
        print("  file_coverage_percent: n/a")

    if stats.approx_chunk_coverage_percent is not None:
        print(
            f"  chunk_coverage_percent: {stats.approx_chunk_coverage_percent:.2f}% "
            "(min(doc_chunks_total, scope_total_chunks) / scope_total_chunks)"
        )
    else:
        print("  chunk_coverage_percent: n/a")


if __name__ == "__main__":
    main()

