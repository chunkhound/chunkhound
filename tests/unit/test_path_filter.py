"""Unit tests for path filter semantics shared by DB and diff search.

Covers the pure string logic of:
  - validate_and_normalize_path_filter() — file vs directory classification
  - path_matches_filter() — in-memory matching (diff search)
  - DuckDBProvider._build_path_like_pattern() — SQL LIKE translation

The last two must agree: a filter has to select the same paths regardless of
whether results come from the DB or from in-memory diff chunks.

No database required — these are pure string functions.
"""

import duckdb
import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.utils.path_filter import (
    path_matches_filter,
    validate_and_normalize_path_filter,
)

# ---------------------------------------------------------------------------
# validate_and_normalize_path_filter
# ---------------------------------------------------------------------------


def _normalize(path: str | None) -> str | None:
    """Shorthand to call the shared normalizer."""
    return validate_and_normalize_path_filter(path)


class TestNormalizeNoneOrEmpty:
    def test_none_returns_none(self) -> None:
        assert _normalize(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _normalize("   ") is None

    @pytest.mark.parametrize("path_filter", ["/", "///", "\\\\"])
    def test_root_filter_returns_none(self, path_filter: str) -> None:
        assert _normalize(path_filter) is None


class TestNormalizeClassifiesDirectories:
    """Paths whose final component has no extension (or is a hidden dir) get
    a trailing slash, making them directory patterns in LIKE queries."""

    def test_plain_directory(self) -> None:
        assert _normalize("src") == "src/"

    def test_nested_directory(self) -> None:
        assert _normalize("src/lib") == "src/lib/"

    def test_hidden_directory(self) -> None:
        assert _normalize(".github") == ".github/"

    def test_hidden_directory_nested(self) -> None:
        assert _normalize(".github/workflows") == ".github/workflows/"

    def test_vscode_directory(self) -> None:
        assert _normalize(".vscode") == ".vscode/"

    def test_leading_slash_stripped(self) -> None:
        assert _normalize("/repo_a") == "repo_a/"

    def test_trailing_slash_preserved(self) -> None:
        assert _normalize("repo_a/") == "repo_a/"


class TestNormalizeClassifiesFiles:
    """Paths whose final component has a non-leading dot are treated as files
    (no trailing slash), producing right-anchored LIKE patterns."""

    def test_python_file(self) -> None:
        assert _normalize("module.py") == "module.py"

    def test_typescript_file(self) -> None:
        assert _normalize("utils.ts") == "utils.ts"

    def test_file_in_directory(self) -> None:
        assert _normalize("src/main.ts") == "src/main.ts"

    def test_no_extension_file(self) -> None:
        # Makefile, Dockerfile, etc. have no dot → treated as directory
        assert _normalize("Makefile") == "Makefile/"

    def test_multi_dot_file(self) -> None:
        assert _normalize("my.file.tar.gz") == "my.file.tar.gz"

    def test_hidden_file_with_known_extension(self) -> None:
        """Hidden files with a recognized extension get file treatment."""
        assert _normalize(".eslintrc.js") == ".eslintrc.js"

    def test_hidden_json_file(self) -> None:
        assert _normalize(".eslintrc.json") == ".eslintrc.json"

    def test_hidden_yml_file(self) -> None:
        assert _normalize(".yamllint.yml") == ".yamllint.yml"

    def test_hidden_file_in_subdirectory(self) -> None:
        assert _normalize("config/.secret.toml") == "config/.secret.toml"


class TestNormalizeHiddenFilesWithoutExtension:
    """Hidden files WITHOUT an extension (.env, .gitignore) start with a dot
    but have no second dot, so they are conservatively classified as directories.
    This is a documented trade-off: the primary use case is directory scoping,
    and source code indexing rarely targets bare dotfiles."""

    def test_env_file_classified_as_directory(self) -> None:
        assert _normalize(".env") == ".env/"

    def test_gitignore_classified_as_directory(self) -> None:
        assert _normalize(".gitignore") == ".gitignore/"

    def test_dockerignore_classified_as_directory(self) -> None:
        assert _normalize(".dockerignore") == ".dockerignore/"


class TestNormalizeBackslashes:
    def test_windows_style(self) -> None:
        assert _normalize("\\repo\\a") == "repo/a/"

    def test_mixed_separators(self) -> None:
        assert _normalize("repo\\a/b") == "repo/a/b/"


class TestNormalizeRejectsDangerous:
    @pytest.mark.parametrize(
        "dangerous",
        [
            "..",
            "~",
            "*",
            "?",
            "[",
            "]",
            "\0",
            "\n",
            "\r",
        ],
    )
    def test_rejects_dangerous_pattern(self, dangerous: str) -> None:
        with pytest.raises(ValueError, match="contains forbidden pattern"):
            _normalize(f"src/{dangerous}/file.py")


# ---------------------------------------------------------------------------
# _build_path_like_pattern
# ---------------------------------------------------------------------------


def _like(path: str) -> str:
    """Shorthand to call the static LIKE builder."""
    return DuckDBProvider._build_path_like_pattern(path)


class TestBuildLikeDirectoryPatterns:
    """Directory patterns (trailing /) get wildcards on both sides."""

    def test_simple_directory(self) -> None:
        assert _like("src/") == "%/src/%"

    def test_nested_directory(self) -> None:
        assert _like("src/lib/") == "%/src/lib/%"

    def test_hidden_directory(self) -> None:
        assert _like(".github/") == "%/.github/%"


class TestBuildLikeFilePatterns:
    """File patterns (no trailing /) are right-anchored to prevent false
    positives like module.py matching module.py.bak."""

    def test_simple_file(self) -> None:
        assert _like("module.py") == "%/module.py"

    def test_file_in_directory(self) -> None:
        assert _like("src/main.ts") == "%/src/main.ts"

    def test_multi_dot_file(self) -> None:
        assert _like("my.file.tar.gz") == "%/my.file.tar.gz"

    def test_hidden_file_with_extension(self) -> None:
        """Hidden files with an extension get right-anchored LIKE patterns."""
        assert _like(".eslintrc.js") == "%/.eslintrc.js"

    def test_hidden_file_in_directory(self) -> None:
        assert _like("config/.secret.toml") == "%/config/.secret.toml"


class TestBuildLikeSpecialChars:
    """Metacharacters in paths are escaped before pattern construction."""

    def test_underscore_is_escaped(self) -> None:
        assert _like("scope_name/") == "%/scope\\_name/%"

    def test_percent_is_escaped(self) -> None:
        assert _like("test%dir/") == "%/test\\%dir/%"


# ---------------------------------------------------------------------------
# path_matches_filter (in-memory equivalent of the SQL LIKE filter)
# ---------------------------------------------------------------------------


def _matches(file_path: str, path_filter: str) -> bool:
    """Normalize a user filter, then match it against a stored path."""
    normalized = validate_and_normalize_path_filter(path_filter)
    assert normalized is not None
    return path_matches_filter(file_path, normalized)


class TestMatchDirectoryFilters:
    def test_matches_file_inside_directory(self) -> None:
        assert _matches("src/main.py", "src")

    def test_matches_nested_file(self) -> None:
        assert _matches("src/lib/deep/util.py", "src/lib")

    def test_matches_directory_anywhere_in_path(self) -> None:
        assert _matches("repo_a/src/main.py", "src")

    def test_rejects_partial_directory_name(self) -> None:
        assert not _matches("src_utils/helper.py", "src")

    def test_rejects_sibling_directory(self) -> None:
        assert not _matches("src/lib2/util.py", "src/lib")

    def test_absolute_stored_path(self) -> None:
        assert _matches("/abs/src/main.py", "src")

    def test_windows_stored_path(self) -> None:
        assert _matches("src\\main.py", "src")


class TestMatchFileFilters:
    """File filters are right-anchored — the m8 regression: a diff-side prefix
    match would treat 'module.py' as a directory and drop every real hit."""

    def test_matches_exact_file(self) -> None:
        assert _matches("src/module.py", "module.py")

    def test_matches_file_at_repo_root(self) -> None:
        assert _matches("module.py", "module.py")

    def test_rejects_suffixed_file(self) -> None:
        assert not _matches("src/module.py.bak", "module.py")

    def test_matches_qualified_path(self) -> None:
        assert _matches("a/src/main.ts", "src/main.ts")

    def test_rejects_different_directory(self) -> None:
        assert not _matches("other/main.ts", "src/main.ts")


class TestMatchAgreesWithLikePattern:
    """In-memory matching must select the same paths as the SQL LIKE pattern.

    Evaluated against real DuckDB LIKE, mirroring the provider's
    ``CONCAT('/', f.path) LIKE ? ESCAPE '\\'`` predicate."""

    @pytest.mark.parametrize(
        ("file_path", "path_filter"),
        [
            ("src/main.py", "src"),
            ("src_utils/helper.py", "src"),
            ("src/lib2/util.py", "src/lib"),
            ("src/module.py", "module.py"),
            ("src/module.py.bak", "module.py"),
            (".github/workflows/ci.yml", ".github"),
            ("config/.eslintrc.js", ".eslintrc.js"),
        ],
    )
    def test_matches_like_semantics(self, file_path: str, path_filter: str) -> None:
        normalized = validate_and_normalize_path_filter(path_filter)
        assert normalized is not None
        pattern = DuckDBProvider._build_path_like_pattern(normalized)
        assert path_matches_filter(file_path, normalized) == _like_matches(
            pattern, "/" + file_path
        )


def _like_matches(pattern: str, value: str) -> bool:
    """Evaluate the provider's LIKE predicate with a real DuckDB engine."""
    with duckdb.connect(":memory:") as conn:
        row = conn.execute("SELECT ? LIKE ? ESCAPE '\\'", [value, pattern]).fetchone()
        assert row is not None
        return bool(row[0])
