"""Language-specific corpus generators for search evaluation.

This module is responsible for:

- Enumerating parser-supported languages
- Building small, idiomatic samples per language that contain a unique token
- Generating additional syntax-heavy samples for parser coverage
- Creating QueryDefinition objects that drive the evaluation harness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE


@dataclass
class QueryDefinition:
    """Definition of a single evaluation query."""

    id: str
    language: Language
    pattern: str  # regex pattern based on unique token
    semantic_query: str  # natural-language query for semantic search
    relevant_paths: list[str]


def build_language_pattern_map() -> dict[Language, str]:
    """Map each parser-supported language to a representative pattern key.

    Uses EXTENSION_TO_LANGUAGE as the single source of truth. Prefers
    extension-based keys (".py") over filename-based keys ("Makefile")
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


def get_supported_languages() -> list[Language]:
    """Return all languages that have parser support."""
    langs = {lang for lang in EXTENSION_TO_LANGUAGE.values()}
    # Deterministic ordering by enum value
    return sorted(langs, key=lambda l: l.value)


def parse_languages_arg(arg: str) -> list[Language]:
    """Parse --languages argument into Language list."""
    supported = get_supported_languages()
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
            "fn main() {\n"
            f"    let marker = \"{token}\";\n"
            "    println!(\"Evaluation marker: {}\", marker);\n"
            "}\n"
        )

    # C style (printf)
    if language == Language.C:
        return (
            "#include <stdio.h>\n\n"
            "int main(void) {\n"
            f"    const char *marker = \"{token}\";\n"
            "    printf(\"Evaluation marker: %s\\n\", marker);\n"
            "    return 0;\n"
            "}\n"
        )

    # C++ style (std::cout)
    if language == Language.CPP:
        return (
            "#include <iostream>\n\n"
            "int main() {\n"
            f"    std::string marker = \"{token}\";\n"
            "    std::cout << \"Evaluation marker: \" << marker << std::endl;\n"
            "    return 0;\n"
            "}\n"
        )

    # Swift style
    if language == Language.SWIFT:
        return (
            "import Foundation\n\n"
            "let marker = \"{token}\"\n".format(token=token)
            + "\nprint(\"Evaluation marker: \\(marker)\")\n"
        )

    # PHP style
    if language == Language.PHP:
        return (
            "<?php\n"
            f"$marker = '{token}';\n"
            "echo 'Evaluation marker: ' . $marker . \"\\n\";\n"
        )

    # Objective-C style
    if language == Language.OBJC:
        return (
            "#import <Foundation/Foundation.h>\n\n"
            "int main(int argc, char *argv[]) {\n"
            "    @autoreleasepool {\n"
            f"        NSString *marker = @\"{token}\";\n"
            "        NSLog(@\"Evaluation marker: %@\", marker);\n"
            "    }\n"
            "    return 0;\n"
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


def build_semantic_query(language: Language) -> str:
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


def create_corpus(
    project_dir: Path, languages: Iterable[Language]
) -> tuple[dict[Language, list[str]], list[QueryDefinition]]:
    """Create evaluation corpus files and query definitions.

    Returns:
        language_to_paths: mapping of language to list of relative file paths
        queries: list of QueryDefinition, one per (language, token)
    """
    language_to_key = build_language_pattern_map()

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
        semantic_query = build_semantic_query(language)

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


