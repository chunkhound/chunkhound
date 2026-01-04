

def test_build_git_pathspecs_ext_and_exact_names():
    from chunkhound.utils import git_discovery as gd

    rel = "pkg/sub"
    includes = ["**/*.py", "**/Makefile", "**/*.ts", "**/*.md"]
    specs = gd.build_git_pathspecs(rel, includes)

    # Should include :(glob) pathspecs for simple extensions and exact names
    assert ":(glob)pkg/sub/**/*.py" in specs
    assert ":(glob)pkg/sub/**/*.ts" in specs
    assert ":(glob)pkg/sub/**/Makefile" in specs
    assert ":(glob)pkg/sub/**/*.md" in specs


def test_build_git_pathspecs_ignores_complex_patterns():
    from chunkhound.utils import git_discovery as gd

    # Complex character classes or negations should not be pushed down
    rel = "src"
    includes = ["**/*[ab].py", "**/*.py"]
    specs = gd.build_git_pathspecs(rel, includes)
    # Only the simple extension gets pushed
    assert ":(glob)src/**/*.py" in specs
    assert all("[ab]" not in s for s in specs)
