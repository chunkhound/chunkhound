"""Shared test stubs for unit tests."""


class _FakeDB:
    """Shared DatabaseProvider stub for unit tests.

    Accepts records as either plain dicts or File model instances.
    When a File model is stored, get_file_by_path converts it to a dict
    automatically (unless as_model=True).
    """

    def __init__(self, records):
        self._records = records  # rel_path -> dict or File model
        self.updated = []
        self._next_id = (
            max(
                [
                    rec.get("id", 0) if isinstance(rec, dict) else (rec.id or 0)
                    for rec in records.values()
                ],
                default=0,
            )
            + 1
        )

    def _to_dict(self, rec):
        """Convert a record to dict if it's a File model."""
        if isinstance(rec, dict):
            return rec
        # Map File model fields to DuckDB column names used by provider queries.
        return {
            "id": rec.id,
            "path": rec.path,
            "size": rec.size_bytes,
            "modified_time": rec.mtime,
            "language": rec.language.value if rec.language else None,
            "content_hash": rec.content_hash,
        }

    def get_file_by_path(self, path: str, as_model: bool = False):
        rec = self._records.get(path)
        if rec is None:
            return None
        if as_model and not isinstance(rec, dict):
            return rec
        return self._to_dict(rec)

    def insert_file(self, file_model):
        """Insert new file and return file_id."""
        file_id = self._next_id
        self._next_id += 1
        rec = {
            "id": file_id,
            "path": file_model.path,
            "size": file_model.size_bytes,
            "modified_time": file_model.mtime,
            "content_hash": file_model.content_hash,
        }
        self._records[file_model.path] = rec
        return file_id

    def update_file(self, file_id: int, **kwargs):
        for key, rec in self._records.items():
            d = self._to_dict(rec)
            if d["id"] == file_id:
                if not isinstance(rec, dict):
                    # Promote File model to dict so updates are visible
                    self._records[key] = d
                    rec = d
                if "content_hash" in kwargs:
                    rec["content_hash"] = kwargs["content_hash"]
                if "size_bytes" in kwargs:
                    rec["size"] = kwargs["size_bytes"]
                if "mtime" in kwargs:
                    rec["modified_time"] = kwargs["mtime"]
                self.updated.append((file_id, kwargs))
                return

    # Transaction stubs
    def begin_transaction(self):
        return None

    def commit_transaction(self, force_checkpoint: bool = False):
        return None

    def rollback_transaction(self):
        return None

    def get_chunks_by_file_id(self, file_id: int, as_model: bool = True):
        return []

    def insert_chunks_batch(self, chunks):
        return []

    def delete_chunks_batch(self, chunk_ids):
        return None

    def has_reclaimable_space(self, operation: str = "") -> bool:
        return False

    def create_deferred_indexes(self) -> None:
        pass

    # Async wrappers
    async def begin_transaction_async(self):
        return self.begin_transaction()

    async def commit_transaction_async(self, force_checkpoint: bool = False):
        return self.commit_transaction(force_checkpoint=force_checkpoint)

    async def rollback_transaction_async(self):
        return self.rollback_transaction()

    async def get_file_by_path_async(self, path: str, as_model: bool = False):
        return self.get_file_by_path(path, as_model)

    async def update_file_async(self, file_id: int, **kwargs):
        return self.update_file(file_id, **kwargs)

    async def insert_file_async(self, file_model):
        return self.insert_file(file_model)

    async def get_chunks_by_file_id_async(self, file_id: int, as_model: bool = False):
        return self.get_chunks_by_file_id(file_id, as_model)

    async def insert_chunks_batch_async(self, chunks):
        return self.insert_chunks_batch(chunks)

    async def delete_chunks_batch_async(self, chunk_ids):
        return self.delete_chunks_batch(chunk_ids)


class _Cfg:
    """Minimal Config stub for IndexingCoordinator tests."""

    class _Indexing:
        cleanup = False
        force_reindex = False
        per_file_timeout_seconds = 0.0
        min_dirs_for_parallel = 4
        max_discovery_workers = 4
        parallel_discovery = False

    indexing = _Indexing()
