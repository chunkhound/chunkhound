from textwrap import dedent

from chunkhound.parsers.yaml_template_sanitizer import (
    SanitizedYaml,
    sanitize_helm_templates,
)


def _prepare(content: str) -> tuple[SanitizedYaml, str]:
    text = dedent(content).lstrip("\n")
    return sanitize_helm_templates(text), text


def test_no_templating_returns_same_text():
    input_text = "apiVersion: v1\nkind: ConfigMap\n"
    sanitized = sanitize_helm_templates(input_text)
    assert sanitized.text == input_text
    assert sanitized.rewrites == []
    assert not sanitized.changed


def test_control_directive_becomes_comment():
    sanitized, original = _prepare(
        """
        {{- if .Values.enabled }}
        kind: Service
        {{- end }}
        """
    )
    assert "# CH_TPL_CTRL:" in sanitized.text
    assert len(sanitized.text.splitlines()) == len(original.splitlines())
    kinds = {rewrite.kind for rewrite in sanitized.rewrites}
    assert {"control"} <= kinds


def test_inline_template_value_becomes_placeholder_scalar():
    sanitized, original = _prepare(
        """
        metadata:
          labels: {{- include "chart.labels" . | nindent 4 }}
        """
    )
    assert "__CH_TPL_MAP__" in sanitized.text
    assert '__CH_TPL_BLOCK__"' in sanitized.text
    sanitized_lines = sanitized.text.splitlines()
    assert sanitized_lines[0] == "metadata:"
    assert sanitized_lines[1].strip() == "labels:"
    assert any(r.kind == "inline_map" for r in sanitized.rewrites)


def test_sequence_item_template_becomes_placeholder():
    sanitized, original = _prepare(
        """
        items:
          - {{ include "chart.tpl" . }}
        """
    )
    assert '- "__CH_TPL_ITEM__"' in sanitized.text
    assert len(sanitized.text.splitlines()) == len(original.splitlines())
    assert any(r.kind == "seq_item" for r in sanitized.rewrites)


def test_template_line_without_key_becomes_comment():
    sanitized, original = _prepare(
        """
        {{ include "chart.tpl" . }}
        """
    )
    assert "# CH_TPL_INCLUDE:" in sanitized.text
    assert len(sanitized.text.splitlines()) == len(original.splitlines())
    assert any(r.kind == "template" for r in sanitized.rewrites)


def test_block_scalar_untouched():
    sanitized, original = _prepare(
        """
        data: |-
          value: {{ .Values.foo }}
        next: ok
        """
    )
    assert "{{ .Values.foo }}" in sanitized.text
    assert any(line.startswith("data: |-") for line in sanitized.text.splitlines())
    assert len(sanitized.text.splitlines()) == len(original.splitlines())


def test_pre_skip_non_yaml_markers():
    # NGINX-style config should be pre-skipped to avoid ryml churn
    text = "server {\n  listen 80;\n}\n"
    sanitized = sanitize_helm_templates(text)
    assert sanitized.pre_skip is True
    assert sanitized.pre_skip_reason == "non_yaml_fragment"


def test_pre_skip_templated_key():
    text = '  "{{ template "common.names.fullname" $ }}-sentinel": 1\n'
    sanitized = sanitize_helm_templates(text)
    assert sanitized.pre_skip is True
    assert sanitized.pre_skip_reason == "templated_key"


def test_pre_skip_block_scalar_threshold(monkeypatch):
    from chunkhound.parsers import yaml_template_sanitizer as ys

    monkeypatch.setattr(ys, "_BLOCK_SCALAR_TPL_THRESHOLD", 2, raising=False)
    text = 'data: |-\n  {{- if .Values.a }}\n  {{- include "x" . }}\n  {{- end }}\n'
    sanitized = ys.sanitize_helm_templates(text)
    assert sanitized.pre_skip is True
    assert sanitized.pre_skip_reason == "block_scalar_template_heavy"
