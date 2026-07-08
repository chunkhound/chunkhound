"""Unit tests for path filter normalization and LIKE pattern construction.

Tests the pure string-transformation logic of:
  - DuckDBProvider._validate_and_normalize_path_filter()
  - DuckDBProvider._build_path_like_pattern()

No database required — these are pure string functions.
"""

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider

# ---------------------------------------------------------------------------
# _validate_and_normalize_path_filter
# ---------------------------------------------------------------------------


def _normalize(path: str | None) -> str | None:
    """Shorthand to call the static normalize method."""
    return DuckDBProvider._validate_and_normalize_path_filter(path)


class TestNormalizeNoneOrEmpty:
    def test_none_returns_none(self) -> None:
        assert _normalize(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _normalize("   ") is None


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
