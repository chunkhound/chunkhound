"""Language-specific tree-sitter mappings for ChunkHound parsers.

This package contains base classes and language-specific implementations
for mapping tree-sitter AST nodes to semantic chunks.
"""

from .base import BaseMapping
from .bash import BashMapping
from .c import CMapping
from .cpp import CppMapping
from .csharp import CSharpMapping
from .css import CssMapping
from .dart import DartMapping
from .elixir import ElixirMapping
from .go import GoMapping
from .groovy import GroovyMapping
from .haskell import HaskellMapping
from .hcl import HclMapping
from .html import HtmlMapping, JinjaMapping
from .java import JavaMapping
from .javascript import JavaScriptMapping
from .json import JsonMapping
from .jsx import JSXMapping
from .kotlin import KotlinMapping
from .lua import LuaMapping
from .makefile import MakefileMapping
from .markdown import MarkdownMapping
from .matlab import MatlabMapping
from .objc import ObjCMapping
from .pdf import PDFMapping
from .php import PHPMapping
from .python import PythonMapping
from .rust import RustMapping
from .scss import ScssMapping
from .sql import SqlMapping
from .svelte import SvelteMapping
from .swift import SwiftMapping
from .text import TextMapping
from .toml import TomlMapping
from .tsx import TSXMapping
from .typescript import TypeScriptMapping
from .vue import VueMapping
from .vue_template import VueTemplateMapping
from .yaml import YamlMapping
from .zig import ZigMapping

__all__ = [
    "BaseMapping",
    "BashMapping",
    "CMapping",
    "CppMapping",
    "CSharpMapping",
    "CssMapping",
    "DartMapping",
    "ElixirMapping",
    "GoMapping",
    "GroovyMapping",
    "HaskellMapping",
    "HclMapping",
    "HtmlMapping",
    "JinjaMapping",
    "JavaMapping",
    "JavaScriptMapping",
    "JsonMapping",
    "JSXMapping",
    "KotlinMapping",
    "LuaMapping",
    "MakefileMapping",
    "MarkdownMapping",
    "MatlabMapping",
    "ObjCMapping",
    "PDFMapping",
    "PHPMapping",
    "PythonMapping",
    "RustMapping",
    "ScssMapping",
    "SqlMapping",
    "SvelteMapping",
    "SwiftMapping",
    "TextMapping",
    "TomlMapping",
    "TSXMapping",
    "TypeScriptMapping",
    "VueMapping",
    "VueTemplateMapping",
    "YamlMapping",
    "ZigMapping",
]
