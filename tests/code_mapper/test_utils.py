from chunkhound.code_mapper.utils import compute_scope_prefix, safe_scope_label


def test_safe_scope_label_normalizes() -> None:
    assert safe_scope_label("scope") == "scope"
    assert safe_scope_label("scope/sub") == "scope_sub"
    assert safe_scope_label("") == "root"
    assert safe_scope_label("/") == "_"


def test_compute_scope_prefix_normalizes() -> None:
    assert compute_scope_prefix("/") is None
    assert compute_scope_prefix("scope") == "scope/"
    assert compute_scope_prefix("scope/") == "scope/"
    assert compute_scope_prefix("scope/sub") == "scope/sub/"
