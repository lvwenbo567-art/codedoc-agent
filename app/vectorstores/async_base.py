from __future__ import annotations

from abc import ABC, abstractmethod

from vectorstores.models import VectorDeleteResult, VectorPoint, VectorSearchFilters, VectorSearchResult, VectorUpsertResult


class AsyncVectorStore(ABC):
    """异步 VectorStore 抽象；与 Day41 同步接口语义一致。"""

    @abstractmethod
    async def ensure_ready(self, *, vector_size: int) -> None: ...

    @abstractmethod
    async def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult: ...

    @abstractmethod
    async def search(self, *, project_id: int, query_vector: list[float], top_k: int,
                     filters: VectorSearchFilters | None = None) -> list[VectorSearchResult]: ...

    @abstractmethod
    async def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult: ...

    @abstractmethod
    async def count(self, *, project_id: int) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...
