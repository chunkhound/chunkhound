from chunkhound.code_mapper.utils import safe_scope_label


def test_safe_scope_label_normalizes() -> None:
    assert safe_scope_label("scope") == "scope"
    assert safe_scope_label("scope/sub") == "scope_sub"
    assert safe_scope_label("") == "root"
    assert safe_scope_label("/") == "_"
