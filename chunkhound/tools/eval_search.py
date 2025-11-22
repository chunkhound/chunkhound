"""Evaluation harness for ChunkHound search.

This tool builds small synthetic corpora that exercise every parser-supported
language, runs regex and semantic searches, and computes simple retrieval
metrics (recall@k/precision@k) plus latency statistics.

Usage (via Makefile target or directly):
    uv run python -m chunkhound.tools.eval_search --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE
from chunkhound.services.directory_indexing_service import DirectoryIndexingService


@dataclass
class QueryDefinition:
    """Definition of a single evaluation query."""

    id: str
    language: Language
    pattern: str  # regex pattern based on unique token
    semantic_query: str  # natural-language query for semantic search
    relevant_paths: list[str]


@dataclass
class QueryMetrics:
    """Per-query evaluation metrics for multiple k values."""

    query_id: str
    language: Language
    pattern: str
    search_type: str
    latency_ms: float
    total_results: int
    first_relevant_rank: int | None
    metrics_by_k: dict[int, dict[str, float]]


@dataclass
class AggregateMetrics:
    """Aggregated metrics across queries."""

    metrics_by_k: dict[int, dict[str, float]]
    latency_stats_ms: dict[str, float]
    mrr: float = 0.0


@dataclass
class EvalResult:
    """Complete evaluation result."""

    mode: str
    search_mode: str
    languages: list[Language]
    ks: list[int]
    per_query: list[QueryMetrics] = field(default_factory=list)
    per_language: dict[str, AggregateMetrics] = field(default_factory=dict)
    global_metrics: AggregateMetrics | None = None


def _build_language_pattern_map() -> dict[Language, str]:
    """Map each parser-supported language to a representative pattern key.

    Uses EXTENSION_TO_LANGUAGE as the single source of truth. Prefers
    extension-based keys (\".py\") over filename-based keys (\"Makefile\")
    when both exist for the same language.
    """
    language_to_key: dict[Language, str] = {}

    for key, lang in EXTENSION_TO_LANGUAGE.items():
        existing = language_to_key.get(lang)
        if existing is None:
            language_to_key[lang] = key
            continue

        # Prefer extension-based keys over filename-based ones
        if not existing.startswith(".") and key.startswith("."):
            language_to_key[lang] = key

    return language_to_key


def _get_supported_languages() -> list[Language]:
    """Return all languages that have parser support."""
    langs = {lang for lang in EXTENSION_TO_LANGUAGE.values()}
    # Deterministic ordering by enum value
    return sorted(langs, key=lambda l: l.value)


def _parse_languages_arg(arg: str) -> list[Language]:
    """Parse --languages argument into Language list."""
    supported = _get_supported_languages()
    if arg.strip().lower() == "all":
        return supported

    value_to_lang = {lang.value.lower(): lang for lang in supported}
    names = [name.strip().lower() for name in arg.split(",") if name.strip()]
    selected: list[Language] = []
    for name in names:
        lang = value_to_lang.get(name)
        if not lang:
            raise ValueError(f"Unknown or unsupported language: {name}")
        selected.append(lang)

    return selected


def _build_language_sample_source(language: Language, token: str) -> str:
    """Generate a minimal code or text snippet for a language.

    The goal is not perfect syntax but to ensure the unique token appears
    in the indexed content in a reasonably idiomatic way, and that the
    snippet's behavior can be described in natural language for semantic
    evaluation.
    """
    # Documentation / text / config styles: describe an "evaluation marker"
    # used for QA/testing so semantic queries can target behavior rather
    # than opaque tokens.
    if language == Language.MARKDOWN:
        return (
            "# Evaluation marker documentation\n\n"
            "This document describes a special evaluation marker used for QA "
            f"checks in automated tests. The marker is `{token}` and it is "
            "referenced by multiple tools during validation.\n"
        )

    if language == Language.JSON:
        return (
            json.dumps(
                {
                    "description": "Configuration for automated QA evaluation",
                    "evaluation_marker": token,
                },
                indent=2,
            )
            + "\n"
        )

    if language == Language.YAML:
        return (
            "description: QA evaluation settings\n"
            f"evaluation_marker: {token}\n"
        )

    if language == Language.TOML:
        return (
            'description = "Configuration for evaluation benchmarks"\n'
            f'evaluation_marker = "{token}"\n'
        )

    if language == Language.HCL:
        return (
            'resource "chunkhound_evaluation" "qa" {\n'
            f'  evaluation_marker = "{token}"\n'
            "}\n"
        )

    if language == Language.TEXT:
        return (
            "This plain text document explains the evaluation marker used during "
            "ChunkHound QA runs. The marker is a short string that tools can "
            f"search for when validating behavior. For this bench it is: {token}\n"
        )

    if language == Language.MAKEFILE:
        return (
            "# Makefile for QA evaluation\n"
            "all:\n"
            f"\t@echo \"Running evaluation with marker: {token}\"\n"
        )

    # Shell / scripting
    if language == Language.BASH:
        return (
            "#!/usr/bin/env bash\n"
            "# Script used in QA evaluation to echo an evaluation marker.\n"
            f"EVAL_MARKER='{token}'\n"
            "echo \"Evaluation marker: ${EVAL_MARKER}\"\n"
        )

    # Python style
    if language == Language.PYTHON:
        return (
            "def eval_search_language_sample() -> str:\n"
            '    """Return an evaluation marker string used for QA checks."""\n'
            f"    return \"{token}\"\n"
        )

    # Haskell style
    if language == Language.HASKELL:
        return (
            "module EvalSearch where\n\n"
            "-- Returns an evaluation marker string used in tests.\n"
            "evaluationMarker :: String\n"
            f"evaluationMarker = \"{token}\"\n"
        )

    # MATLAB style
    if language == Language.MATLAB:
        return (
            "% MATLAB function for QA evaluation\n"
            "% Returns an evaluation marker string used in tests.\n"
            "function out = eval_search_language_sample()\n"
            f"  out = '{token}';\n"
            "end\n"
        )

    # Vue single-file component
    if language == Language.VUE:
        return (
            "<template>\n"
            f"  <div>Evaluation marker: {token}</div>\n"
            "</template>\n"
            "<script>\n"
            "export default {\n"
            "  name: 'EvalSearchSample',\n"
            "  // Displays an evaluation marker used in QA examples.\n"
            "};\n"
            "</script>\n"
        )

    # Markdown-like text for PDF is handled separately as bytes.

    # Groovy style
    if language == Language.GROOVY:
        return (
            "// Groovy script used in QA evaluation.\n"
            "// Prints an evaluation marker using println.\n"
            f"def marker = '{token}'\n"
            "println \"Evaluation marker: ${marker}\"\n"
        )

    # Kotlin style
    if language == Language.KOTLIN:
        return (
            "// Kotlin program used in QA evaluation.\n"
            "// Prints an evaluation marker using println.\n"
            "fun main() {\n"
            f"    val marker = \"{token}\"\n"
            "    println(\"Evaluation marker: $marker\")\n"
            "}\n"
        )

    # JavaScript style (console.log based)
    if language in {
        Language.JAVASCRIPT,
        Language.JSX,
        Language.TSX,
    }:
        return (
            "// Script used in QA evaluation to log an evaluation marker.\n"
            "function logEvaluationMarker() {\n"
            f"  const marker = '{token}';\n"
            "  console.log('Evaluation marker:', marker);\n"
            "}\n"
        )

    # TypeScript style (typed function + console.log)
    if language == Language.TYPESCRIPT:
        return (
            "// TypeScript function used in QA evaluation.\n"
            "// Logs an evaluation marker with a typed variable.\n"
            "function logEvaluationMarker(marker: string): void {\n"
            "  console.log('Evaluation marker:', marker);\n"
            "}\n"
            f"const marker: string = '{token}';\n"
            "logEvaluationMarker(marker);\n"
        )

    # Java style (System.out.println in main)
    if language == Language.JAVA:
        return (
            "public class EvalSearchSample {\n"
            "    // Prints an evaluation marker used in QA runs.\n"
            "    public static void main(String[] args) {\n"
            f"        String marker = \"{token}\";\n"
            "        System.out.println(\"Evaluation marker: \" + marker);\n"
            "    }\n"
            "}\n"
        )

    # C# style (Console.WriteLine in Main)
    if language == Language.CSHARP:
        return (
            "using System;\n\n"
            "public class EvalSearchSample {\n"
            "    // Prints an evaluation marker used in QA runs.\n"
            "    public static void Main(string[] args) {\n"
            f"        var marker = \"{token}\";\n"
            "        Console.WriteLine($\"Evaluation marker: {marker}\");\n"
            "    }\n"
            "}\n"
        )

    # Go style (fmt.Println in main)
    if language == Language.GO:
        return (
            "package main\n\n"
            "import \"fmt\"\n\n"
            "// Prints an evaluation marker used in QA runs.\n"
            "func main() {\n"
            f"    marker := \"{token}\"\n"
            "    fmt.Println(\"Evaluation marker:\", marker)\n"
            "}\n"
        )

    # Rust style (println! in main)
    if language == Language.RUST:
        return (
            "// Rust program used in QA evaluation.\n"
            "// Prints an evaluation marker to stdout.\n"
            "fn main() {\n"
            f"    let marker = \"{token}\";\n"
            "    println!(\"Evaluation marker: {}\", marker);\n"
            "}\n"
        )

    # Swift style (print in main)
    if language == Language.SWIFT:
        return (
            "// Swift program used in QA evaluation.\n"
            "// Prints an evaluation marker using print().\n"
            f"let marker = \"{token}\"\n"
            "print(\"Evaluation marker: \\(marker)\")\n"
        )

    # Zig style (std.debug.print in main)
    if language == Language.ZIG:
        return (
            "const std = @import(\"std\");\n\n"
            "pub fn main() !void {\n"
            f"    const marker = \"{token}\";\n"
            "    try std.io.getStdOut().writer().print(\n"
            "        \"Evaluation marker: {s}\\n\",\n"
            "        .{marker},\n"
            "    );\n"
            "}\n"
        )

    # Objective-C style (@implementation with NSLog)
    if language == Language.OBJC:
        return (
            "// Objective-C implementation used in QA evaluation.\n"
            "#import <Foundation/Foundation.h>\n\n"
            "@interface EvalSearchSample : NSObject\n"
            "+ (void)printMarker;\n"
            "@end\n\n"
            "@implementation EvalSearchSample\n"
            "+ (void)printMarker {\n"
            f"    NSString *marker = @\"{token}\";\n"
            "    NSLog(@\"Evaluation marker: %@\", marker);\n"
            "}\n"
            "@end\n\n"
            "int main(int argc, char *argv[]) {\n"
            "    @autoreleasepool {\n"
            "        [EvalSearchSample printMarker];\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    # PHP style (requires proper PHP syntax for parser)
    if language == Language.PHP:
        return (
            "<?php\n"
            "// PHP function used in QA evaluation.\n"
            "// Returns an evaluation marker string.\n"
            f"function eval_search_sample() {{\n"
            f"    return '{token}';\n"
            "}\n"
        )

    # Default C-style comment-based snippet for most remaining languages
    c_style_languages = {
        Language.JAVA,
        Language.CSHARP,
        Language.TYPESCRIPT,
        Language.JAVASCRIPT,
        Language.TSX,
        Language.JSX,
        Language.GROOVY,
        Language.KOTLIN,
        Language.GO,
        Language.RUST,
        Language.ZIG,
        Language.C,
        Language.CPP,
        Language.OBJC,
        Language.SWIFT,
    }

    if language in c_style_languages:
        return (
            "// Function used in QA evaluation.\n"
            "// It prints an evaluation marker to standard output.\n"
            "int main(void) {\n"
            f"  const char* marker = \"{token}\";\n"
            "  printf(\"Evaluation marker: %s\\n\", marker);\n"
            "  return 0;\n"
            "}\n"
        )

    # Fallback: plain text with token
    return f"{language.value} sample\n{token}\n"


def _build_language_syntax_samples(language: Language) -> dict[str, str]:
    """Build additional syntax-heavy samples per language.

    These files are used only to enrich the bench corpus for parser coverage.
    They deliberately DO NOT contain the unique evaluation token so they do
    not change relevance labels for retrieval metrics.
    """
    samples: dict[str, str] = {}

    if language == Language.PYTHON:
        samples["syntax_showcase.py"] = (
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "from typing import Any, Callable, Generic, Iterable, TypeVar\n\n"
            "T = TypeVar(\"T\")\n\n"
            "@dataclass\n"
            "class EvalConfig(Generic[T]):\n"
            "    name: str\n"
            "    values: list[T]\n\n"
            "    def filter(self, predicate: Callable[[T], bool]) -> list[T]:\n"
            "        return [v for v in self.values if predicate(v)]\n\n"
            "async def _aiter(items: Iterable[int]):\n"
            "    for item in items:\n"
            "        yield item\n\n"
            "async def async_collect(n: int) -> list[int]:\n"
            "    return [i async for i in _aiter(range(n))]\n\n"
            "def pattern_match(value: int) -> str:\n"
            "    match value:\n"
            "        case 0:\n"
            "            return \"zero\"\n"
            "        case 1 | 2:\n"
            "            return \"small\"\n"
            "        case _:\n"
            "            return \"other\"\n"
        )

    if language in {Language.JAVASCRIPT, Language.JSX, Language.TSX}:
        samples["syntax_showcase.js"] = (
            "// Modern JavaScript syntax showcase for parser coverage.\n"
            "export class SearchClient {\n"
            "  #endpoint;\n"
            "  constructor(endpoint) {\n"
            "    this.#endpoint = endpoint ?? \"http://localhost\";\n"
            "  }\n\n"
            "  async search(query, opts = {}) {\n"
            "    const params = { q: query, ...opts };\n"
            "    const url = `${this.#endpoint}/search`;\n"
            "    const res = await fetch(url, { method: \"POST\", body: JSON.stringify(params) });\n"
            "    return res?.ok ? res.json() : [];\n"
            "  }\n"
            "}\n\n"
            "export const mapResults = (items) =>\n"
            "  items.flatMap((item, index) => ({ ...item, index }));\n"
        )

    if language == Language.TYPESCRIPT:
        samples["syntax_showcase.ts"] = (
            "// TypeScript syntax showcase with generics and enums.\n"
            "export enum RankSignal {\n"
            "  Low = 0,\n"
            "  Medium = 1,\n"
            "  High = 2,\n"
            "}\n\n"
            "export interface RankedResult<T> {\n"
            "  value: T;\n"
            "  score: number;\n"
            "  signal: RankSignal;\n"
            "}\n\n"
            "export function rerank<T>(values: T[], score: (v: T) => number): RankedResult<T>[] {\n"
            "  return values\n"
            "    .map((v) => ({ value: v, score: score(v), signal: RankSignal.Medium }))\n"
            "    .sort((a, b) => b.score - a.score);\n"
            "}\n"
        )

    if language == Language.JAVA:
        samples["SyntaxShowcase.java"] = (
            "import java.util.List;\n"
            "import java.util.Map;\n"
            "import java.util.stream.Collectors;\n\n"
            "public class SyntaxShowcase {\n"
            "    public record Entry(String key, int value) {}\n\n"
            "    public static Map<String, Integer> aggregate(List<Entry> entries) {\n"
            "        return entries.stream()\n"
            "            .filter(e -> e.value() > 0)\n"
            "            .collect(Collectors.groupingBy(Entry::key,\n"
            "                Collectors.summingInt(Entry::value)));\n"
            "    }\n"
            "}\n"
        )

    if language == Language.CSHARP:
        samples["SyntaxShowcase.cs"] = (
            "using System;\n"
            "using System.Collections.Generic;\n"
            "using System.Linq;\n\n"
            "public static class SyntaxShowcase {\n"
            "    public static IEnumerable<(string key, int count)> CountKeys(\n"
            "        IEnumerable<string> keys)\n"
            "    {\n"
            "        return from k in keys\n"
            "               group k by k into g\n"
            "               select (g.Key, g.Count());\n"
            "    }\n"
            "}\n"
        )

    if language == Language.GO:
        samples["syntax_showcase.go"] = (
            "package main\n\n"
            "import \"sort\"\n\n"
            "type Pair struct {\n"
            "    Key   string\n"
            "    Value int\n"
            "}\n\n"
            "func SortPairs(pairs []Pair) {\n"
            "    sort.Slice(pairs, func(i, j int) bool {\n"
            "        if pairs[i].Value == pairs[j].Value {\n"
            "            return pairs[i].Key < pairs[j].Key\n"
            "        }\n"
            "        return pairs[i].Value > pairs[j].Value\n"
            "    })\n"
            "}\n"
        )

    if language == Language.RUST:
        samples["syntax_showcase.rs"] = (
            "use std::collections::HashMap;\n\n"
            "pub fn aggregate_counts(items: &[String]) -> HashMap<String, usize> {\n"
            "    let mut map = HashMap::new();\n"
            "    for item in items {\n"
            "        *map.entry(item.clone()).or_insert(0) += 1;\n"
            "    }\n"
            "    map\n"
            "}\n"
        )

    if language == Language.KOTLIN:
        samples["SyntaxShowcase.kt"] = (
            "data class Metric(val name: String, val value: Int)\n\n"
            "fun topMetrics(metrics: List<Metric>): List<Metric> =\n"
            "    metrics.filter { it.value > 0 }\n"
            "        .sortedByDescending { it.value }\n"
            "        .take(10)\n"
        )

    if language == Language.SWIFT:
        samples["SyntaxShowcase.swift"] = (
            "struct Metric {\n"
            "    let name: String\n"
            "    let value: Int\n"
            "}\n\n"
            "func topMetrics(_ metrics: [Metric]) -> [Metric] {\n"
            "    return metrics\n"
            "        .filter { $0.value > 0 }\n"
            "        .sorted { $0.value > $1.value }\n"
            "}\n"
        )

    return samples


def _build_minimal_pdf_bytes(token: str) -> bytes:
    """Create a minimal PDF file containing the token as text."""
    # Very small, valid-enough PDF with one page and a text stream describing
    # the evaluation marker for semantic search.
    text = f"Evaluation marker used in QA benchmarks: {token}"
    pdf_text = "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pdf_text += "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    content_stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
    stream_len = len(content_stream)
    pdf_text += (
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R >>\nendobj\n"
    )
    pdf_text += (
        f"4 0 obj\n<< /Length {stream_len} >>\n"
        "stream\n"
        f"{content_stream}\n"
        "endstream\nendobj\n"
    )
    pdf_text += "xref\n0 5\n0000000000 65535 f \n"
    pdf_text += "trailer\n<< /Root 1 0 R /Size 5 >>\n"
    pdf_text += "startxref\n0\n%%EOF\n"
    return pdf_text.encode("utf-8")


def _create_corpus(
    project_dir: Path, languages: Iterable[Language]
) -> tuple[dict[Language, list[str]], list[QueryDefinition]]:
    """Create evaluation corpus files and query definitions.

    Returns:
        language_to_paths: mapping of language to list of relative file paths
        queries: list of QueryDefinition, one per (language, token)
    """
    language_to_key = _build_language_pattern_map()

    language_to_paths: dict[Language, list[str]] = {}
    queries: list[QueryDefinition] = []

    for language in languages:
        key = language_to_key.get(language)
        if key is None:
            logger.debug(f"Skipping language without pattern mapping: {language.value}")
            continue

        subdir = Path("eval_lang") / language.value
        subdir_path = project_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        if key.startswith("."):
            relative_path = subdir / f"sample{key}"
        else:
            relative_path = subdir / key

        file_path = project_dir / relative_path
        token = f"{language.value}_qa_unique"
        semantic_query = _build_semantic_query(language)

        if language == Language.PDF:
            file_path.write_bytes(_build_minimal_pdf_bytes(token))
        else:
            content = _build_language_sample_source(language, token)
            file_path.write_text(content, encoding="utf-8")

            # Add additional syntax-heavy samples for parser coverage (no token).
            extra_samples = _build_language_syntax_samples(language)
            for extra_name, extra_content in extra_samples.items():
                extra_path = subdir_path / extra_name
                extra_path.write_text(extra_content, encoding="utf-8")

        rel_str = str(relative_path).replace("\\", "/")
        language_to_paths.setdefault(language, []).append(rel_str)

        query_id = f"{language.value}_unique_token"
        queries.append(
            QueryDefinition(
                id=query_id,
                language=language,
                pattern=token,
                semantic_query=semantic_query,
                relevant_paths=[rel_str],
            )
        )

    return language_to_paths, queries


def _build_semantic_query(language: Language) -> str:
    """Build a natural-language query for semantic evaluation.

    Queries describe behavior or purpose rather than referencing tokens
    or filenames. They are designed to be realistic but still tied to the
    small per-language snippets generated in this module.
    """
    if language == Language.PYTHON:
        return (
            "Python function named eval_search_language_sample that returns the "
            "evaluation marker string used for QA checks in the evaluation harness"
        )
    if language == Language.BASH:
        return (
            "shell script that echoes an evaluation marker string in a QA run"
        )
    if language == Language.C:
        return (
            "C program that uses printf to print an evaluation marker to stdout"
        )
    if language == Language.CPP:
        return (
            "C++ program that prints an evaluation marker message to standard output"
        )
    if language == Language.JAVA:
        return (
            "Java class with a main method that calls System.out.println to print "
            "an evaluation marker"
        )
    if language == Language.CSHARP:
        return (
            "C# program whose Main method calls Console.WriteLine to print an "
            "evaluation marker"
        )
    if language == Language.GO:
        return (
            "Go program whose main function uses fmt.Println to print an "
            "evaluation marker"
        )
    if language == Language.RUST:
        return (
            "Rust program whose main function uses the println! macro to print an "
            "evaluation marker"
        )
    if language == Language.SWIFT:
        return (
            "Swift code that calls print to display an evaluation marker string"
        )
    if language in {Language.JAVASCRIPT, Language.JSX, Language.TSX}:
        return (
            "JavaScript code that calls console.log to print an evaluation marker "
            "to the console"
        )
    if language == Language.TYPESCRIPT:
        return (
            "TypeScript function with a typed string parameter that calls "
            "console.log to print an evaluation marker"
        )
    if language == Language.MAKEFILE:
        return (
            "Makefile with a default target that prints an evaluation marker when run"
        )
    if language == Language.JSON:
        return "configuration file defining an evaluation_marker field for automated tests"
    if language == Language.YAML:
        return "YAML configuration that stores an evaluation marker used in QA"
    if language == Language.TOML:
        return "TOML configuration describing settings for evaluation benchmarks"
    if language == Language.MARKDOWN:
        return "documentation explaining an evaluation marker used during QA"
    if language == Language.TEXT:
        return "plain text document that explains the evaluation marker used for testing"
    if language == Language.HCL:
        return "infrastructure configuration resource that includes an evaluation marker"
    if language == Language.VUE:
        return "single-file component that renders an evaluation marker in the template"
    if language == Language.MATLAB:
        return (
            "MATLAB function named eval_search_language_sample that returns an "
            "evaluation marker string for tests"
        )
    if language == Language.PHP:
        return "PHP function that returns an evaluation marker string used in QA"
    if language == Language.HASKELL:
        return "Haskell definition that provides an evaluation marker string for tests"
    if language == Language.GROOVY:
        return (
            "Groovy script that defines a marker variable and uses println to "
            "print the evaluation marker"
        )
    if language == Language.KOTLIN:
        return (
            "Kotlin program with a main function that calls println to print an "
            "evaluation marker"
        )
    if language == Language.OBJC:
        return (
            "Objective-C program with an @implementation that uses NSLog to print "
            "an evaluation marker"
        )
    if language == Language.PDF:
        return "PDF document that describes the evaluation marker used in QA benchmarks"

    # Default for other programming languages (Java, C#, Go, Rust, etc.)
    return (
        "small program used in QA evaluation that prints an evaluation marker"
    )


def _build_config(project_dir: Path, config_path: str | None) -> Config:
    """Construct Config for the evaluation run.

    When a config file is provided, it is passed via a minimal argparse-style
    namespace so Config can apply its usual precedence rules. The database path
    is always overridden to live inside the temporary project directory.
    """
    args: Any | None = None
    if config_path:
        args = SimpleNamespace(config=config_path, path=str(project_dir))

    config = Config(args=args, target_dir=project_dir)

    db_path = project_dir / ".chunkhound" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config.database.path = db_path

    return config


async def _index_corpus(
    project_dir: Path,
    config: Config,
    db_path: Path,
    with_embeddings: bool,
) -> None:
    """Index the corpus using the standard service stack.

    When with_embeddings is True, missing embeddings are generated using the
    configured embedding provider. This is required for semantic evaluation
    (search_mode=semantic).
    """
    services = create_services(db_path=db_path, config=config)

    indexing_service = DirectoryIndexingService(
        indexing_coordinator=services.indexing_coordinator,
        config=config,
        progress_callback=lambda msg: logger.debug(f"[index] {msg}"),
        progress=None,
    )

    stats = await indexing_service.process_directory(
        project_dir,
        no_embeddings=not with_embeddings,
    )

    if stats.errors_encountered:
        logger.warning(
            f"Indexing completed with {len(stats.errors_encountered)} errors"
        )


async def _run_queries(
    config: Config,
    db_path: Path,
    queries: list[QueryDefinition],
    ks: list[int],
    search_mode: str,
) -> list[QueryMetrics]:
    """Run queries and collect per-query metrics."""
    services = create_services(db_path=db_path, config=config)
    search_service = services.search_service

    per_query_metrics: list[QueryMetrics] = []
    max_k = max(ks) if ks else 10

    for query in queries:
        start = time.perf_counter()
        if search_mode == "semantic":
            search_text = query.semantic_query
            results, _ = await search_service.search_semantic(
                query=search_text,
                page_size=max_k,
                offset=0,
            )
            search_type = "semantic"
        else:
            search_text = query.pattern
            results, _ = await search_service.search_regex_async(
                pattern=search_text,
                page_size=max_k,
                offset=0,
                path_filter=None,
            )
            search_type = "regex"

        latency_ms = (time.perf_counter() - start) * 1000.0

        result_paths = [r.get("file_path") for r in results if r.get("file_path")]
        relevant = set(query.relevant_paths)

        first_rank: int | None = None
        for idx, path in enumerate(result_paths, start=1):
            if path in relevant:
                first_rank = idx
                break

        metrics_by_k: dict[int, dict[str, float]] = {}

        for k in ks:
            top_k = result_paths[:k]
            hits = len(relevant.intersection(top_k))
            total_relevant = len(relevant)

            recall = float(hits) / float(total_relevant) if total_relevant else 0.0
            if k > 0:
                precision = float(hits) / float(min(k, len(result_paths))) if result_paths else 0.0
            else:
                precision = 0.0

            metrics_by_k[k] = {
                "recall": recall,
                "precision": precision,
                "hit_count": float(hits),
            }

        per_query_metrics.append(
            QueryMetrics(
                query_id=query.id,
                language=query.language,
                pattern=search_text,
                search_type=search_type,
                latency_ms=latency_ms,
                total_results=len(result_paths),
                first_relevant_rank=first_rank,
                metrics_by_k=metrics_by_k,
            )
        )

    return per_query_metrics


def _aggregate_metrics(
    per_query: list[QueryMetrics],
    ks: list[int],
) -> AggregateMetrics:
    """Aggregate metrics across a set of queries."""
    if not per_query:
        return AggregateMetrics(
            metrics_by_k={
                k: {"recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "ndcg": 0.0}
                for k in ks
            },
            latency_stats_ms={"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            mrr=0.0,
        )

    # Latency stats
    latencies = [q.latency_ms for q in per_query]
    latencies_sorted = sorted(latencies)
    mean_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_index = int(0.95 * (len(latencies_sorted) - 1))
    p95 = latencies_sorted[p95_index]
    max_latency = max(latencies_sorted)

    # Mean reciprocal rank (MRR)
    reciprocals: list[float] = []
    for q in per_query:
        rank = q.first_relevant_rank
        if rank is not None and rank > 0:
            reciprocals.append(1.0 / float(rank))
    mrr = statistics.mean(reciprocals) if reciprocals else 0.0

    metrics_by_k: dict[int, dict[str, float]] = {}

    for k in ks:
        recalls: list[float] = []
        precisions: list[float] = []
        hits_flags: list[float] = []
        ndcgs: list[float] = []
        for q in per_query:
            m = q.metrics_by_k.get(k)
            if not m:
                continue
            recalls.append(m.get("recall", 0.0))
            precisions.append(m.get("precision", 0.0))
            hit_count = m.get("hit_count", 0.0)
            hits_flags.append(1.0 if hit_count > 0 else 0.0)

            # nDCG@k for binary relevance with at most one relevant document per query.
            rank = q.first_relevant_rank
            if rank is not None and rank <= k:
                dcg = 1.0 / math.log2(float(rank) + 1.0)
                idcg = 1.0 / math.log2(2.0)
                ndcg = dcg / idcg if idcg > 0.0 else 0.0
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)

        if recalls:
            avg_recall = statistics.mean(recalls)
            avg_precision = statistics.mean(precisions)
            hit_rate = statistics.mean(hits_flags)
            avg_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0
        else:
            avg_recall = 0.0
            avg_precision = 0.0
            hit_rate = 0.0
            avg_ndcg = 0.0

        metrics_by_k[k] = {
            "recall": avg_recall,
            "precision": avg_precision,
            "hit_rate": hit_rate,
            "ndcg": avg_ndcg,
        }

    latency_stats: dict[str, float] = {
        "mean": mean_latency,
        "p50": p50,
        "p95": p95,
        "max": max_latency,
    }

    return AggregateMetrics(
        metrics_by_k=metrics_by_k,
        latency_stats_ms=latency_stats,
        mrr=mrr,
    )


async def _run_mode_mixed(
    languages: list[Language],
    ks: list[int],
    search_mode: str,
    config_path: str | None,
    bench_root: Path | None,
) -> EvalResult:
    """Run evaluation with a single mixed-language corpus."""
    if bench_root is not None:
        project_dir = bench_root
        project_dir.mkdir(parents=True, exist_ok=True)

        # Build or refresh corpus in persistent bench directory
        _, queries = _create_corpus(project_dir, languages)

        config = _build_config(project_dir, config_path)
        db_path = config.database.path

        logger.info(
            f"Mixed-mode evaluation: {len(languages)} languages, "
            f"{len(queries)} queries, db={db_path}"
        )

        if search_mode == "semantic":
            errors = config.validate_for_command("search")
            if errors:
                raise RuntimeError(
                    "Semantic search requires a configured embedding provider.\n"
                    + "\n".join(errors)
                )

        await _index_corpus(
            project_dir,
            config,
            db_path,
            with_embeddings=(search_mode == "semantic"),
        )
        per_query = await _run_queries(config, db_path, queries, ks, search_mode)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir(parents=True, exist_ok=True)

            # Build corpus
            _, queries = _create_corpus(project_dir, languages)

            # Configure database inside project directory
            config = _build_config(project_dir, config_path)
            db_path = config.database.path

            logger.info(
                f"Mixed-mode evaluation: {len(languages)} languages, "
                f"{len(queries)} queries, db={db_path}"
            )

            if search_mode == "semantic":
                errors = config.validate_for_command("search")
                if errors:
                    raise RuntimeError(
                        "Semantic search requires a configured embedding provider.\n"
                        + "\n".join(errors)
                    )

            await _index_corpus(
                project_dir,
                config,
                db_path,
                with_embeddings=(search_mode == "semantic"),
            )
            per_query = await _run_queries(config, db_path, queries, ks, search_mode)

    # Aggregate metrics after temp directory cleanup
    per_language: dict[str, AggregateMetrics] = {}
    for language in languages:
        lang_queries = [q for q in per_query if q.language == language]
        per_language[language.value] = _aggregate_metrics(lang_queries, ks)

    global_metrics = _aggregate_metrics(per_query, ks)

    return EvalResult(
        mode="mixed",
        search_mode=search_mode,
        languages=languages,
        ks=ks,
        per_query=per_query,
        per_language=per_language,
        global_metrics=global_metrics,
    )


async def _run_mode_per_language(
    languages: list[Language],
    ks: list[int],
    search_mode: str,
    config_path: str | None,
    bench_root: Path | None,
) -> EvalResult:
    """Run evaluation with one corpus per language."""
    all_queries: list[QueryMetrics] = []
    per_language: dict[str, AggregateMetrics] = {}

    for language in languages:
        if bench_root is not None:
            project_dir = bench_root
            project_dir.mkdir(parents=True, exist_ok=True)

            _, queries = _create_corpus(project_dir, [language])

            config = _build_config(project_dir, config_path)
            db_path = config.database.path

            logger.info(
                f"Per-language evaluation (bench): {language.value}, "
                f"{len(queries)} queries, db={db_path}"
            )

            if search_mode == "semantic":
                errors = config.validate_for_command("search")
                if errors:
                    raise RuntimeError(
                        "Semantic search requires a configured embedding provider.\n"
                        + "\n".join(errors)
                    )

            await _index_corpus(
                project_dir,
                config,
                db_path,
                with_embeddings=(search_mode == "semantic"),
            )
            per_query = await _run_queries(config, db_path, queries, ks, search_mode)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                project_dir = Path(tmp) / "project"
                project_dir.mkdir(parents=True, exist_ok=True)

                _, queries = _create_corpus(project_dir, [language])

                config = _build_config(project_dir, config_path)
                db_path = config.database.path

                logger.info(
                    f"Per-language evaluation: {language.value}, "
                    f"{len(queries)} queries, db={db_path}"
                )

                if search_mode == "semantic":
                    errors = config.validate_for_command("search")
                    if errors:
                        raise RuntimeError(
                            "Semantic search requires a configured embedding provider.\n"
                            + "\n".join(errors)
                        )

                await _index_corpus(
                    project_dir,
                    config,
                    db_path,
                    with_embeddings=(search_mode == "semantic"),
                )
                per_query = await _run_queries(
                    config, db_path, queries, ks, search_mode
                )

        all_queries.extend(per_query)
        per_language[language.value] = _aggregate_metrics(per_query, ks)

    global_metrics = _aggregate_metrics(all_queries, ks)

    return EvalResult(
        mode="per-language",
        search_mode=search_mode,
        languages=languages,
        ks=ks,
        per_query=all_queries,
        per_language=per_language,
        global_metrics=global_metrics,
    )


def _format_human_summary(result: EvalResult) -> None:
    """Print a concise human-readable summary of evaluation metrics."""
    if result.global_metrics is None:
        print("No metrics computed.")
        return

    languages_str = ", ".join(lang.value for lang in result.languages)
    print(f"Mode: {result.mode}, search={result.search_mode}")
    print(f"Languages: {languages_str}")
    print(f"Queries: {len(result.per_query)}")

    print("\nGlobal metrics:")
    for k in sorted(result.ks):
        m = result.global_metrics.metrics_by_k.get(k, {})
        recall = m.get("recall", 0.0)
        precision = m.get("precision", 0.0)
        hit_rate = m.get("hit_rate", 0.0)
        ndcg = m.get("ndcg", 0.0)
        print(
            f"  k={k:2d}: recall={recall:.3f}, "
            f"precision={precision:.3f}, hit-rate={hit_rate:.3f}, ndcg={ndcg:.3f}"
        )

    lat = result.global_metrics.latency_stats_ms
    print(
        f"\nLatency (ms): mean={lat['mean']:.1f}, "
        f"p50={lat['p50']:.1f}, p95={lat['p95']:.1f}, max={lat['max']:.1f}"
    )

    print(f"\nMRR: {result.global_metrics.mrr:.3f}")

    print("\nPer-language metrics:")
    for language in sorted(result.languages, key=lambda l: l.value):
        lang_metrics = result.per_language.get(language.value)
        if not lang_metrics:
            continue
        line_parts = [f"  {language.value}:"]
        for k in sorted(result.ks):
            m = lang_metrics.metrics_by_k.get(k, {})
            recall = m.get("recall", 0.0)
            hit_rate = m.get("hit_rate", 0.0)
            ndcg = m.get("ndcg", 0.0)
            line_parts.append(f"k={k}: r={recall:.2f}, h={hit_rate:.2f}, n={ndcg:.2f}")
        print(" ".join(line_parts))


def _build_json_payload(result: EvalResult) -> dict[str, Any]:
    """Convert EvalResult to JSON-serializable payload."""
    global_metrics = (
        {
            "metrics_by_k": result.global_metrics.metrics_by_k,
            "latency_ms": result.global_metrics.latency_stats_ms,
            "mrr": result.global_metrics.mrr,
        }
        if result.global_metrics
        else None
    )

    per_language: dict[str, Any] = {}
    for lang, metrics in result.per_language.items():
        per_language[lang] = {
            "metrics_by_k": metrics.metrics_by_k,
            "latency_ms": metrics.latency_stats_ms,
            "mrr": metrics.mrr,
        }

    per_query_payload: list[dict[str, Any]] = []
    for q in result.per_query:
        per_query_payload.append(
            {
                "id": q.query_id,
                "language": q.language.value,
                "pattern": q.pattern,
                "search_type": q.search_type,
                "latency_ms": q.latency_ms,
                "total_results": q.total_results,
                "first_relevant_rank": q.first_relevant_rank,
                "metrics_by_k": q.metrics_by_k,
            }
        )

    return {
        "mode": result.mode,
        "search_mode": result.search_mode,
        "languages": [lang.value for lang in result.languages],
        "ks": result.ks,
        "global": global_metrics,
        "per_language": per_language,
        "per_query": per_query_payload,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local evaluation harness for ChunkHound search.\n"
            "Builds small synthetic corpora per language, runs regex or semantic "
            "queries, and computes recall@k/precision@k and latency metrics."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["mixed", "per-language"],
        default="mixed",
        help="Evaluation mode: mixed corpus or one corpus per language (default: mixed)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["regex", "semantic"],
        default="regex",
        help="Search type to evaluate (default: regex).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help="Comma-separated list of languages by enum value (or 'all', default).",
    )
    parser.add_argument(
        "--k",
        dest="ks",
        type=int,
        action="append",
        default=None,
        help="Top-k values to evaluate (can be repeated, default: 1,5,10).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a config file for embedding/indexing settings.",
    )
    parser.add_argument(
        "--bench-id",
        type=str,
        default=None,
        help="Optional benchmark ID. When provided, corpus is stored under "
        ".chunkhound/benches/<bench-id>/ instead of a temporary directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON metrics report.",
    )

    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        languages = _parse_languages_arg(args.languages)
    except ValueError as e:
        logger.error(str(e))
        return 1

    ks: list[int] = sorted(set(args.ks or [1, 5, 10]))
    if not ks:
        logger.error("No k values specified for evaluation.")
        return 1

    search_mode = args.search_mode
    config_path = args.config
    bench_root: Path | None = None
    if args.bench_id:
        bench_root = (
            Path.cwd() / ".chunkhound" / "benches" / args.bench_id / "source"
        )

    try:
        if args.mode == "mixed":
            result = await _run_mode_mixed(
                languages, ks, search_mode, config_path, bench_root
            )
        else:
            result = await _run_mode_per_language(
                languages, ks, search_mode, config_path, bench_root
            )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

    _format_human_summary(result)

    if args.output:
        payload = _build_json_payload(result)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics report to {output_path}")

    return 0


def main() -> None:
    """Entry point for python -m chunkhound.tools.eval_search."""
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
