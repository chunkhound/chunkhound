"""Language-specific tree-sitter mappings for ChunkHound parsers.

This package contains base classes and language-specific implementations
for mapping tree-sitter AST nodes to semantic chunks.
"""

from .base import BaseMapping
from .bash import BashMapping
from .c import CMapping
from .cpp import CppMapping
from .csharp import CSharpMapping
from .dart import DartMapping
from .go import GoMapping
from .groovy import GroovyMapping
from .haskell import HaskellMapping
from .hcl import HclMapping
from .java import JavaMapping
from .javascript import JavaScriptMapping
from .json import JsonMapping
from .jsx import JSXMapping
from .kotlin import KotlinMapping
from .makefile import MakefileMapping
from .markdown import MarkdownMapping
from .matlab import MatlabMapping
from .objc import ObjCMapping
from .pdf import PDFMapping
from .php import PHPMapping
from .python import PythonMapping
from .rust import RustMapping
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
    "DartMapping",
    "GoMapping",
    "GroovyMapping",
    "HaskellMapping",
    "JavaMapping",
    "JavaScriptMapping",
    "JsonMapping",
    "JSXMapping",
    "KotlinMapping",
    "HclMapping",
    "MakefileMapping",
    "MarkdownMapping",
    "MatlabMapping",
    "ObjCMapping",
    "PDFMapping",
    "PHPMapping",
    "PythonMapping",
    "RustMapping",
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
