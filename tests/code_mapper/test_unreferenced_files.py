
from chunkhound.code_mapper.coverage import compute_unreferenced_scope_files


class _Provider:
    def __init__(self) -> None:
        self._files = ["scope/a.py", "scope/b.py", "scope/c.py"]

    def get_scope_file_paths(self, scope_prefix: str | None) -> list[str]:
        if scope_prefix == "scope/":
            return list(self._files)
        return []


class _Services:
    def __init__(self) -> None:
        self.provider = _Provider()


def test_compute_unreferenced_scope_files_filters_referenced() -> None:
    services = _Services()
    referenced = ["scope/a.py", "scope/b.py"]

    unreferenced = compute_unreferenced_scope_files(
        services=services,
        scope_label="scope",
        referenced_files=referenced,
    )

    assert unreferenced == ["scope/c.py"]
