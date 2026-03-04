"""Tests for Python import path resolution via resolve_import_paths()."""

import pytest

from chunkhound.parsers.mappings.python import PythonMapping


class TestPythonImportResolution:
    """Test Python import path resolution via resolve_import_paths()."""

    # =========================================================================
    # A. Single Imports (with and without "from")
    # =========================================================================

    def test_single_import_file(self, tmp_path):
        """Test `import foo` resolves to foo.py."""
        mapping = PythonMapping()

        foo_file = tmp_path / "foo.py"
        foo_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import foo"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_single_import_package(self, tmp_path):
        """Test `import foo` resolves to foo/__init__.py when package exists."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "foo"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import foo"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == init_file

    def test_single_from_import_symbol(self, tmp_path):
        """Test `from foo import bar` resolves to foo.py when bar is a symbol."""
        mapping = PythonMapping()

        foo_file = tmp_path / "foo.py"
        foo_file.write_text("bar = 42")

        source_file = tmp_path / "main.py"
        import_text = "from foo import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_single_from_import_submodule(self, tmp_path):
        """Test `from foo import bar` resolves to foo/bar.py when bar is a submodule."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "foo"
        pkg_dir.mkdir()
        bar_file = pkg_dir / "bar.py"
        bar_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from foo import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == bar_file

    # =========================================================================
    # B. Imports with Namespaces (a.b.c) and without (a)
    # =========================================================================

    def test_nested_module_import(self, tmp_path):
        """Test `import a.b.c` resolves to a/b/c.py."""
        mapping = PythonMapping()

        a_dir = tmp_path / "a"
        a_dir.mkdir()
        b_dir = a_dir / "b"
        b_dir.mkdir()
        c_file = b_dir / "c.py"
        c_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a.b.c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == c_file

    def test_nested_package_import(self, tmp_path):
        """Test `import a.b.c` resolves to a/b/c/__init__.py when c is a package."""
        mapping = PythonMapping()

        a_dir = tmp_path / "a"
        a_dir.mkdir()
        b_dir = a_dir / "b"
        b_dir.mkdir()
        c_dir = b_dir / "c"
        c_dir.mkdir()
        init_file = c_dir / "__init__.py"
        init_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a.b.c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == init_file

    def test_nested_from_import(self, tmp_path):
        """Test `from a.b.c import d` resolves to a/b/c.py when d is a symbol."""
        mapping = PythonMapping()

        a_dir = tmp_path / "a"
        a_dir.mkdir()
        b_dir = a_dir / "b"
        b_dir.mkdir()
        c_file = b_dir / "c.py"
        c_file.write_text("d = 42")

        source_file = tmp_path / "main.py"
        import_text = "from a.b.c import d"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == c_file

    def test_flat_import(self, tmp_path):
        """Test `import foo` resolves to foo.py (flat namespace)."""
        mapping = PythonMapping()

        foo_file = tmp_path / "foo.py"
        foo_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import foo"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    # =========================================================================
    # C. Multiple Imports on Same Line vs Single
    # =========================================================================

    def test_multi_import_direct(self, tmp_path):
        """Test `import a, b, c` resolves to all three files."""
        mapping = PythonMapping()

        a_file = tmp_path / "a.py"
        a_file.write_text("")
        b_file = tmp_path / "b.py"
        b_file.write_text("")
        c_file = tmp_path / "c.py"
        c_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a, b, c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 3
        assert set(resolved) == {a_file, b_file, c_file}

    def test_multi_import_nested(self, tmp_path):
        """Test `import a.b, x.y` resolves to nested modules."""
        mapping = PythonMapping()

        a_dir = tmp_path / "a"
        a_dir.mkdir()
        ab_file = a_dir / "b.py"
        ab_file.write_text("")

        x_dir = tmp_path / "x"
        x_dir.mkdir()
        xy_file = x_dir / "y.py"
        xy_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a.b, x.y"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {ab_file, xy_file}

    def test_multi_import_partial(self, tmp_path):
        """Test `import a, b, c` resolves only existing modules."""
        mapping = PythonMapping()

        a_file = tmp_path / "a.py"
        a_file.write_text("")
        b_file = tmp_path / "b.py"
        b_file.write_text("")
        # c.py does not exist

        source_file = tmp_path / "main.py"
        import_text = "import a, b, c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    # =========================================================================
    # D. Import Symbol from Package vs File
    # =========================================================================

    def test_from_import_symbols_from_file(self, tmp_path):
        """Test `from pkg import Cls, func` resolves to pkg.py when symbols are in file."""
        mapping = PythonMapping()

        pkg_file = tmp_path / "pkg.py"
        pkg_file.write_text("class Cls: pass\ndef func(): pass")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import Cls, func"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_file

    def test_from_import_submodules_from_package(self, tmp_path):
        """Test `from pkg import a, b` resolves to submodules when they exist."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import a, b"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    def test_from_import_mixed_submod_and_symbol(self, tmp_path):
        """Test `from pkg import a, Cls` resolves submodule and base file for symbol."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        pkg_init = pkg_dir / "__init__.py"
        pkg_init.write_text("class Cls: pass")
        a_file = pkg_dir / "a.py"
        a_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import a, Cls"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, pkg_init}

    # =========================================================================
    # E. Imports with Parentheses (single/multiple symbols)
    # =========================================================================

    def test_from_import_parens_single(self, tmp_path):
        """Test `from pkg import (name)` resolves correctly."""
        mapping = PythonMapping()

        pkg_file = tmp_path / "pkg.py"
        pkg_file.write_text("name = 42")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (name)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_file

    def test_from_import_parens_multi(self, tmp_path):
        """Test `from pkg import (a, b, c)` resolves all submodules."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")
        c_file = pkg_dir / "c.py"
        c_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (a, b, c)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 3
        assert set(resolved) == {a_file, b_file, c_file}

    def test_from_import_parens_submod_and_symbol(self, tmp_path):
        """Test `from pkg import (sub, Cls)` resolves submodule and base for symbol."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        pkg_init = pkg_dir / "__init__.py"
        pkg_init.write_text("class Cls: pass")
        sub_file = pkg_dir / "sub.py"
        sub_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (sub, Cls)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {sub_file, pkg_init}

    # =========================================================================
    # F. Multi-line Imports
    # =========================================================================

    def test_multiline_from_import(self, tmp_path):
        """Test multi-line from import with parentheses."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (\n    a,\n    b\n)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    def test_multiline_from_import_with_comments(self, tmp_path):
        """Test multi-line from import handles inline comments gracefully."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (\n    a,  # module a\n    b  # module b\n)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    def test_multiline_from_import_trailing_comma(self, tmp_path):
        """Test multi-line from import with trailing comma."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import (\n    a,\n    b,\n)"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    def test_from_import_star(self, tmp_path):
        """Test `from pkg import *` resolves to pkg/__init__.py or pkg.py."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        pkg_init = pkg_dir / "__init__.py"
        pkg_init.write_text("__all__ = ['a', 'b']")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import *"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_init

    def test_from_import_star_file(self, tmp_path):
        """Test `from pkg import *` resolves to pkg.py when it's a file."""
        mapping = PythonMapping()

        pkg_file = tmp_path / "pkg.py"
        pkg_file.write_text("__all__ = ['foo', 'bar']")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import *"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_file

    # =========================================================================
    # G. Imports with Aliases ("as")
    # =========================================================================

    def test_single_import_alias(self, tmp_path):
        """Test `import foo as bar` resolves to foo.py."""
        mapping = PythonMapping()

        foo_file = tmp_path / "foo.py"
        foo_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import foo as bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_multi_import_all_aliases(self, tmp_path):
        """Test `import a as x, b as y` resolves both modules."""
        mapping = PythonMapping()

        a_file = tmp_path / "a.py"
        a_file.write_text("")
        b_file = tmp_path / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a as x, b as y"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    def test_multi_import_mixed_aliases(self, tmp_path):
        """Test `import a, b as y, c` resolves all three modules."""
        mapping = PythonMapping()

        a_file = tmp_path / "a.py"
        a_file.write_text("")
        b_file = tmp_path / "b.py"
        b_file.write_text("")
        c_file = tmp_path / "c.py"
        c_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a, b as y, c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 3
        assert set(resolved) == {a_file, b_file, c_file}

    def test_from_import_alias_single(self, tmp_path):
        """Test `from pkg import name as alias` resolves to pkg.py."""
        mapping = PythonMapping()

        pkg_file = tmp_path / "pkg.py"
        pkg_file.write_text("name = 42")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import name as alias"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_file

    def test_from_import_alias_multi(self, tmp_path):
        """Test `from pkg import a as x, b as y` resolves both submodules."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import a as x, b as y"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    # =========================================================================
    # H. Relative Imports (.a.b) vs Absolute
    # =========================================================================

    def test_relative_import_single_dot_module(self, tmp_path):
        """Test `from .foo import bar` resolves to sibling foo.py."""
        mapping = PythonMapping()

        # Create package structure: pkg/sub/source.py importing from .foo
        pkg_dir = tmp_path / "pkg" / "sub"
        pkg_dir.mkdir(parents=True)
        foo_file = pkg_dir / "foo.py"
        foo_file.write_text("bar = 42")

        source_file = pkg_dir / "source.py"
        import_text = "from .foo import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_relative_import_single_dot_submodule(self, tmp_path):
        """Test `from .pkg import bar` resolves to sibling pkg/bar.py."""
        mapping = PythonMapping()

        # Create package structure: main/source.py and main/pkg/bar.py
        main_dir = tmp_path / "main"
        main_dir.mkdir()
        pkg_dir = main_dir / "pkg"
        pkg_dir.mkdir()
        bar_file = pkg_dir / "bar.py"
        bar_file.write_text("")

        source_file = main_dir / "source.py"
        import_text = "from .pkg import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == bar_file

    def test_relative_import_dot_only(self, tmp_path):
        """Test `from . import foo` resolves to sibling foo.py."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        foo_file = pkg_dir / "foo.py"
        foo_file.write_text("")

        source_file = pkg_dir / "source.py"
        import_text = "from . import foo"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_relative_import_double_dot(self, tmp_path):
        """Test `from .. import utils` resolves to parent utils.py."""
        mapping = PythonMapping()

        # Create: pkg/utils.py and pkg/sub/source.py
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        utils_file = pkg_dir / "utils.py"
        utils_file.write_text("")

        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir()
        source_file = sub_dir / "source.py"
        import_text = "from .. import utils"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == utils_file

    def test_relative_import_double_dot_module(self, tmp_path):
        """Test `from ..foo import bar` resolves to parent's foo.py."""
        mapping = PythonMapping()

        # Create: pkg/foo.py and pkg/sub/source.py
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        foo_file = pkg_dir / "foo.py"
        foo_file.write_text("bar = 42")

        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir()
        source_file = sub_dir / "source.py"
        import_text = "from ..foo import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == foo_file

    def test_relative_import_multi(self, tmp_path):
        """Test `from ..pkg import a, b, c` resolves multiple submodules."""
        mapping = PythonMapping()

        # Create: root/sub/pkg/{a,b,c}.py and root/sub/deep/source.py
        # From deep/source.py, `..` goes up to sub/, then resolves `pkg` there
        root = tmp_path / "root"
        root.mkdir()
        sub_dir = root / "sub"
        sub_dir.mkdir()
        pkg_dir = sub_dir / "pkg"
        pkg_dir.mkdir()
        a_file = pkg_dir / "a.py"
        a_file.write_text("")
        b_file = pkg_dir / "b.py"
        b_file.write_text("")
        c_file = pkg_dir / "c.py"
        c_file.write_text("")

        deep_dir = sub_dir / "deep"
        deep_dir.mkdir()
        source_file = deep_dir / "source.py"
        import_text = "from ..pkg import a, b, c"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 3
        assert set(resolved) == {a_file, b_file, c_file}

    def test_relative_import_too_many_dots(self, tmp_path):
        """Test relative import with too many dots returns empty."""
        mapping = PythonMapping()

        # Source file at pkg/source.py trying to go up 5 levels
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        source_file = pkg_dir / "source.py"
        import_text = "from ..... import foo"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert resolved == []

    def test_relative_import_nonexistent(self, tmp_path):
        """Test relative import to nonexistent module returns empty."""
        mapping = PythonMapping()

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        source_file = pkg_dir / "source.py"
        import_text = "from .nonexistent import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert resolved == []

    def test_relative_import_dotted_path(self, tmp_path):
        """Test `from .pkg.sub import x` resolves through dotted path."""
        mapping = PythonMapping()

        # Create: main/pkg/sub/x.py and main/source.py
        main_dir = tmp_path / "main"
        main_dir.mkdir()
        pkg_dir = main_dir / "pkg"
        pkg_dir.mkdir()
        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir()
        x_file = sub_dir / "x.py"
        x_file.write_text("")

        source_file = main_dir / "source.py"
        import_text = "from .pkg.sub import x"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == x_file

    def test_absolute_import(self, tmp_path):
        """Test `from pkg import bar` resolves to pkg.py for absolute import."""
        mapping = PythonMapping()

        pkg_file = tmp_path / "pkg.py"
        pkg_file.write_text("bar = 42")

        source_file = tmp_path / "main.py"
        import_text = "from pkg import bar"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 1
        assert resolved[0] == pkg_file

    # =========================================================================
    # I. Mix of import-from-file and import-from-package
    # =========================================================================

    def test_mixed_file_and_package_imports(self, tmp_path):
        """Test `import a, pkg.b` resolves both file and nested module."""
        mapping = PythonMapping()

        a_file = tmp_path / "a.py"
        a_file.write_text("")

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        b_file = pkg_dir / "b.py"
        b_file.write_text("")

        source_file = tmp_path / "main.py"
        import_text = "import a, pkg.b"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert len(resolved) == 2
        assert set(resolved) == {a_file, b_file}

    # =========================================================================
    # J. Failure Cases
    # =========================================================================

    def test_external_package(self, tmp_path):
        """Test `import os` returns empty (external packages not resolved)."""
        mapping = PythonMapping()

        source_file = tmp_path / "main.py"
        import_text = "import os"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert resolved == []

    def test_nonexistent_module(self, tmp_path):
        """Test `import nonexistent` returns empty."""
        mapping = PythonMapping()

        source_file = tmp_path / "main.py"
        import_text = "import nonexistent"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert resolved == []

    def test_all_unresolvable(self, tmp_path):
        """Test `import x, y, z` returns empty when none exist."""
        mapping = PythonMapping()

        source_file = tmp_path / "main.py"
        import_text = "import x, y, z"

        resolved = mapping.resolve_import_paths(import_text, tmp_path, source_file)
        assert resolved == []
