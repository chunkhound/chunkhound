from __future__ import annotations

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.tools.eval.language_samples import (
    QueryDefinition,
    create_corpus,
    parse_languages_arg,
)


def test_parse_languages_arg_single_language() -> None:
    langs = parse_languages_arg("python")
    assert Language.PYTHON in langs
    assert len(langs) == 1


def test_parse_languages_arg_all() -> None:
    langs = parse_languages_arg("all")
    # At least one language should be supported
    assert langs
    assert isinstance(langs[0], Language)


def test_create_corpus_creates_files_and_queries(tmp_path: Path) -> None:
    project_dir = tmp_path
    languages = [Language.PYTHON, Language.JSON]

    language_to_paths, queries = create_corpus(project_dir, languages)

    # One query per language
    assert len(queries) == len(languages)

    paths_by_language = {lang: paths for lang, paths in language_to_paths.items()}
    for language in languages:
        # Each requested language should have at least one file path
        assert language in paths_by_language
        rel_paths = paths_by_language[language]
        assert rel_paths

        # Files should exist on disk under the project directory
        for rel in rel_paths:
            file_path = project_dir / rel
            assert file_path.is_file()

    # Query definitions should reference existing files
    path_set = {
        (project_dir / rel).resolve()
        for paths in language_to_paths.values()
        for rel in paths
    }
    assert isinstance(queries[0], QueryDefinition)
    for query in queries:
        for rel in query.relevant_paths:
            target = (project_dir / rel).resolve()
            assert target in path_set

