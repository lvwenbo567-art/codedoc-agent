from __future__ import annotations

from typing import Any
from urllib.parse import urlparse#用来判断 Qdrant 是否是本机服务：

from vectorstores.async_base import AsyncVectorStore
from vectorstores.base import VectorDimensionMismatchError
from vectorstores.models import VectorDeleteResult, VectorPoint, VectorSearchFilters, VectorSearchResult, VectorUpsertResult


class AsyncQdrantVectorStore(AsyncVectorStore):
    """使用 AsyncQdrantClient 的非阻塞 Qdrant 适配器。"""

    def __init__(self, *, url: str, collection_name: str, api_key: str | None = None,
                 prefer_grpc: bool = False, timeout_seconds: float = 10, client: Any | None = None) -> None:
        self.collection_name = collection_name
        if client is not None:
            self.client = client
            self.models = None
            return
        from qdrant_client import AsyncQdrantClient, models
        self.models = models
        local = urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}
        self.client = AsyncQdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc,
                                        timeout=timeout_seconds, **({"trust_env": False} if local else {}))

    async def ensure_ready(self, *, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")
        exists = await self.client.collection_exists(self.collection_name)
        if exists:
            collection = await self.client.get_collection(self.collection_name)
            vectors = collection.config.params.vectors
            size = getattr(vectors, "size", None)
            if size is not None and int(size) != vector_size:
                raise VectorDimensionMismatchError(f"Qdrant Collection 维度 {size} 与目标维度 {vector_size} 不一致")
            await self._ensure_payload_indexes()
            return
        if self.models is None:
            await self.client.create_collection(collection_name=self.collection_name,
                                                vectors_config={"size": vector_size, "distance": "Cosine"})
        else:
            await self.client.create_collection(collection_name=self.collection_name,
                                                vectors_config=self.models.VectorParams(size=vector_size, distance=self.models.Distance.COSINE))
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        if self.models is None:
            return
        for name in ("project_id", "chunk_type", "source_path", "source_suffix", "content_hash", "embedding_model"):
            schema = self.models.PayloadSchemaType.INTEGER if name == "project_id" else self.models.PayloadSchemaType.KEYWORD
            try:
                await self.client.create_payload_index(collection_name=self.collection_name, field_name=name, field_schema=schema, wait=True)
            except Exception:
                continue

    def _build_filter(self, *, project_id: int, filters: VectorSearchFilters | None) -> Any:
        if self.models is None:
            return {"must": [{"key": "project_id", "match": {"value": project_id}}]}
        must = [self.models.FieldCondition(key="project_id", match=self.models.MatchValue(value=project_id))]
        if filters:
            for key in ("chunk_type", "source_path", "source_suffix", "content_hash", "embedding_model"):
                value = getattr(filters, key)
                if value is not None:
                    must.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)))
            if filters.chunk_ids:
                must.append(self.models.FieldCondition(key="chunk_id", match=self.models.MatchAny(any=list(filters.chunk_ids))))
        return self.models.Filter(must=must)

    async def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if not points:
            return VectorUpsertResult(0, 0, 0)
        await self.ensure_ready(vector_size=points[0].dimension)
        for point in points:
            if point.dimension != points[0].dimension:
                raise VectorDimensionMismatchError("同一批 Point 的向量维度不一致")
        batches = 0
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            if self.models is None:
                payload = [{"id": p.point_id, "vector": p.vector, "payload": p.payload} for p in batch]
            else:
                payload = [self.models.PointStruct(id=p.point_id, vector=p.vector, payload=p.payload) for p in batch]
            await self.client.upsert(collection_name=self.collection_name, points=payload, wait=True)
            batches += 1
        return VectorUpsertResult(len(points), len(points), batches)

    async def search(self, *, project_id: int, query_vector: list[float], top_k: int,
                     filters: VectorSearchFilters | None = None) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        response = await self.client.query_points(collection_name=self.collection_name, query=query_vector,
                                                  query_filter=self._build_filter(project_id=project_id, filters=filters),
                                                  limit=top_k, with_payload=True, with_vectors=False)
        points = getattr(response, "points", response)
        results = []
        for item in points:
            payload = dict(getattr(item, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or "")
            if chunk_id:
                results.append(VectorSearchResult(str(getattr(item, "id", "")), project_id, chunk_id,
                                                  float(getattr(item, "score", 0.0)), payload))
        return results

    async def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult:
        normalized = [str(item).strip() for item in chunk_ids if str(item).strip()]
        if not normalized:
            return VectorDeleteResult(0, 0)
        await self.client.delete(collection_name=self.collection_name,
                                 points_selector=self._build_filter(project_id=project_id, filters=VectorSearchFilters(chunk_ids=tuple(normalized))), wait=True)
        return VectorDeleteResult(None, len(normalized))

    async def count(self, *, project_id: int) -> int:
        result = await self.client.count(collection_name=self.collection_name,
                                         count_filter=self._build_filter(project_id=project_id, filters=None), exact=True)
        return int(getattr(result, "count", result))

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result
