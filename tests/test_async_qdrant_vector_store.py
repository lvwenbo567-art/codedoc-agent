from __future__ import annotations

from types import SimpleNamespace

import pytest

from vectorstores.async_qdrant_vector_store import AsyncQdrantVectorStore
from vectorstores.models import VectorPoint


class FakeAsyncQdrant:
    def __init__(self) -> None:
        self.exists = False
        self.points = []
        self.closed = False
    async def collection_exists(self, _): return self.exists
    async def create_collection(self, **_): self.exists = True
    async def get_collection(self, _): return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2))) )
    async def upsert(self, *, points, **_): self.points.extend(points)
    async def query_points(self, **_):
        point = SimpleNamespace(id="p", score=0.9, payload={"chunk_id": "c1", "project_id": 1})
        return SimpleNamespace(points=[point])
    async def count(self, **_): return SimpleNamespace(count=1)
    async def delete(self, **_): return None
    async def close(self): self.closed = True


@pytest.mark.asyncio
async def test_async_qdrant_upsert_search_count_and_close() -> None:
    fake = FakeAsyncQdrant()
    store = AsyncQdrantVectorStore(url="http://unused", collection_name="c", client=fake)
    point = VectorPoint(point_id="p", project_id=1, chunk_id="c1", vector=[1.0, 0.0], payload={"chunk_id": "c1", "project_id": 1})
    result = await store.upsert(points=[point])
    assert result.upserted_count == 1 and fake.exists
    assert (await store.search(project_id=1, query_vector=[1.0, 0.0], top_k=1))[0].chunk_id == "c1"
    assert await store.count(project_id=1) == 1
    await store.close()
    assert fake.closed
