from chunkhound.autodoc.markdown_utils import _extract_description


def test_extract_description_strips_ordered_list_marker() -> None:
    markdown = "\n".join(
        [
            "1. First item",
            "2. Second item",
        ]
    )

    assert _extract_description(markdown) == "First item"

