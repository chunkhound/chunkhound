"""Parsing time must grow roughly linearly with file size (#561).

Large generated Go files (e.g. Kubernetes' 4 MB zz_generated.openapi.go) stalled
indexing because per-chunk work re-processed the whole file. Comparing two sizes
keeps the check independent of machine speed.
"""

import time

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


def _generated_go(functions: int) -> str:
    parts = ['package openapi\n\nimport common "k8s.io/kube-openapi/pkg/common"\n']
    for i in range(functions):
        parts.append(
            f"""
// schema_T{i} returns the OpenAPI definition for T{i}.
func schema_T{i}(ref common.ReferenceCallback) common.OpenAPIDefinition {{
	return common.OpenAPIDefinition{{
		Schema: spec.Schema{{
			SchemaProps: spec.SchemaProps{{
				Description: "T{i} is a generated type.",
				Type:        []string{{"object"}},
				Required:    []string{{"kind", "metadata"}},
			}},
		}},
		Dependencies: []string{{"k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"}},
	}}
}}
"""
        )
    return "".join(parts)


def _parse_seconds(functions: int) -> float:
    parser = ParserFactory().create_parser(Language.GO)
    content = _generated_go(functions)
    start = time.perf_counter()
    chunks = parser.parse_content(content, None, FileId(1))
    elapsed = time.perf_counter() - start
    assert len(chunks) >= functions
    return elapsed


def test_large_go_file_parse_time_scales_linearly():
    _parse_seconds(50)  # warm up the parser and tree-sitter query compilation
    small = _parse_seconds(400)
    large = _parse_seconds(3200)

    # 8x the content: linear work takes ~8x as long, quadratic work ~64x.
    ratio = large / small
    assert ratio < 20, (
        f"Parsing 8x more Go took {ratio:.1f}x longer "
        f"({small:.2f}s -> {large:.2f}s); expected roughly linear growth."
    )
