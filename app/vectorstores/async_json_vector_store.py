from __future__ import annotations

import asyncio

from vectorstores.async_base import AsyncVectorStore
from vectorstores.json_vector_store import JsonVectorStore
from vectorstores.models import VectorDeleteResult, VectorPoint, VectorSearchFilters, VectorSearchResult, VectorUpsertResult


class AsyncJsonVectorStore(AsyncVectorStore):
    """用 to_thread 复用现有 JSON 存储，避免阻塞 FastAPI 事件循环。"""

    def __init__(self, *, store: JsonVectorStore) -> None:
        self.store = store

    async def ensure_ready(self, *, vector_size: int) -> None:
        await asyncio.to_thread(self.store.ensure_ready, vector_size=vector_size)

    async def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult:
        return await asyncio.to_thread(self.store.upsert, points=points, batch_size=batch_size)

    async def search(self, *, project_id: int, query_vector: list[float], top_k: int,
                     filters: VectorSearchFilters | None = None) -> list[VectorSearchResult]:
        return await asyncio.to_thread(self.store.search, project_id=project_id, query_vector=query_vector, top_k=top_k, filters=filters)

    async def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult:
        return await asyncio.to_thread(self.store.delete_chunks, project_id=project_id, chunk_ids=chunk_ids)

    async def count(self, *, project_id: int) -> int:
        return await asyncio.to_thread(self.store.count, project_id=project_id)

    async def close(self) -> None:
        await asyncio.to_thread(self.store.close)
