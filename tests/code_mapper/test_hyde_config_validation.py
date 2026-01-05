import pytest

from chunkhound.code_mapper.models import HydeConfig


def test_hyde_config_allows_zero_caps_for_snippets() -> None:
    cfg = HydeConfig(
        max_scope_files=0,
        max_snippet_files=0,
        max_snippet_chars=0,
        max_completion_tokens=1,
        max_snippet_tokens=1,
    )
    assert cfg.max_scope_files == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_scope_files", -1),
        ("max_snippet_files", -1),
        ("max_snippet_chars", -1),
        ("max_completion_tokens", 0),
        ("max_snippet_tokens", 0),
    ],
)
def test_hyde_config_rejects_invalid_values(field: str, value: int) -> None:
    kwargs = {
        "max_scope_files": 1,
        "max_snippet_files": 0,
        "max_snippet_chars": 0,
        "max_completion_tokens": 1,
        "max_snippet_tokens": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        HydeConfig(**kwargs)
