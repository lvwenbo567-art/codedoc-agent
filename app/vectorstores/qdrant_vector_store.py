from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from vectorstores.base import (
    VectorDimensionMismatchError,
    VectorStore,
    VectorStoreConfigurationError,
)
from vectorstores.models import (
    VectorDeleteResult,
    VectorPoint,
    VectorSearchFilters,
    VectorSearchResult,
    VectorUpsertResult,
)

PAYLOAD_INDEX_FIELDS = (
    "project_id",
    "chunk_type",
    "source_path",
    "source_suffix",
    "content_hash",
    "embedding_model",
)


class QdrantVectorStore(VectorStore):
    """基于 Qdrant 的 VectorStore。qdrant-client 未安装时会给出明确配置错误。"""

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout_seconds: float = 10,
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name

        if client is not None:
            self.client = client
            self.models = None
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "使用 QdrantVectorStore 需要安装 qdrant-client：python -m pip install qdrant-client"
            ) from exc

        self.models = models
        parsed_url = urlparse(url)
        is_local_qdrant = parsed_url.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        http_client_options = (
            {"trust_env": False}
            if is_local_qdrant
            else {}
        )
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout_seconds,
            **http_client_options,
        )

    def _distance(self) -> Any:
        if self.models is None:
            return "Cosine"

        return self.models.Distance.COSINE

    def ensure_ready(self, *, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")

        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            collection = None

        if collection is not None:
            vectors_config = getattr(collection.config.params, "vectors", None)
            size = getattr(vectors_config, "size", None)

            if size is not None and int(size) != vector_size:
                raise VectorDimensionMismatchError(
                    f"Qdrant Collection 维度 {size} 与目标维度 {vector_size} 不一致"
                )

            self._ensure_payload_indexes()
            return

        if self.models is None:
            vectors_config = {
                "size": vector_size,
                "distance": "Cosine",
            }
        else:
            vectors_config = self.models.VectorParams(
                size=vector_size,
                distance=self.models.Distance.COSINE,
            )

        create_collection = getattr(
            self.client,
            "create_collection",
            None,
        )

        if callable(create_collection):
            create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )
        else:
            # 兼容 Day41 测试中只实现 recreate_collection 的 Fake Client。
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        if self.models is None:
            return

        for field_name in PAYLOAD_INDEX_FIELDS:
            schema = (
                self.models.PayloadSchemaType.INTEGER
                if field_name == "project_id"
                else self.models.PayloadSchemaType.KEYWORD
            )
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                # 索引已存在或测试假客户端不支持时，不阻塞主流程。
                continue

    def _build_filter(self, *, project_id: int, filters: VectorSearchFilters | None) -> Any:
        if self.models is None:
            must = [{"key": "project_id", "match": {"value": project_id}}]
            for key in ("chunk_type", "source_path", "source_suffix", "content_hash", "embedding_model"):
                value = getattr(filters, key, None) if filters else None
                if value is not None:
                    must.append({"key": key, "match": {"value": value}})
            if filters and filters.chunk_ids:
                must.append({"key": "chunk_id", "match": {"any": list(filters.chunk_ids)}})
            return {"must": must}

        must = [
            self.models.FieldCondition(
                key="project_id",
                match=self.models.MatchValue(value=project_id),
            )
        ]

        if filters is not None:
            for key in ("chunk_type", "source_path", "source_suffix", "content_hash", "embedding_model"):
                value = getattr(filters, key)
                if value is not None:
                    must.append(
                        self.models.FieldCondition(
                            key=key,
                            match=self.models.MatchValue(value=value),
                        )
                    )
            if filters.chunk_ids:
                must.append(
                    self.models.FieldCondition(
                        key="chunk_id",
                        match=self.models.MatchAny(any=list(filters.chunk_ids)),
                    )
                )

        return self.models.Filter(must=must)

    def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if not points:
            return VectorUpsertResult(received_count=0, upserted_count=0, batch_count=0)

        self.ensure_ready(vector_size=points[0].dimension)
        batch_count = 0

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            batch_count += 1

            if self.models is None:
                qdrant_points = [
                    {"id": point.point_id, "vector": point.vector, "payload": point.payload}
                    for point in batch
                ]
            else:
                qdrant_points = [
                    self.models.PointStruct(
                        id=point.point_id,
                        vector=point.vector,
                        payload=point.payload,
                    )
                    for point in batch
                ]

            self.client.upsert(collection_name=self.collection_name, points=qdrant_points)

        return VectorUpsertResult(
            received_count=len(points),
            upserted_count=len(points),
            batch_count=batch_count,
        )

    def search(
        self,
        *,
        project_id: int,
        query_vector: list[float],
        top_k: int,
        filters: VectorSearchFilters | None = None,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_filter = self._build_filter(project_id=project_id, filters=filters)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            raw_points = getattr(response, "points", response)
        else:
            raw_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        results: list[VectorSearchResult] = []

        for item in raw_points:
            payload = dict(getattr(item, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or "")
            results.append(
                VectorSearchResult(
                    point_id=str(getattr(item, "id", "")),
                    project_id=int(payload.get("project_id", project_id)),
                    chunk_id=chunk_id,
                    score=float(getattr(item, "score", 0.0)),
                    payload=payload,
                )
            )

        return results

    def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult:
        if not chunk_ids:
            return VectorDeleteResult(deleted_count=0, requested_count=0)

        query_filter = self._build_filter(
            project_id=project_id,
            filters=VectorSearchFilters(chunk_ids=tuple(chunk_ids)),
        )
        self.client.delete(collection_name=self.collection_name, points_selector=query_filter)
        return VectorDeleteResult(deleted_count=None, requested_count=len(chunk_ids))

    def delete_project(self, *, project_id: int) -> VectorDeleteResult:
        query_filter = self._build_filter(project_id=project_id, filters=None)
        self.client.delete(collection_name=self.collection_name, points_selector=query_filter)
        return VectorDeleteResult(deleted_count=None, requested_count=None)

    def count(self, *, project_id: int) -> int:
        query_filter = self._build_filter(project_id=project_id, filters=None)
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(getattr(result, "count", result))

    def list_chunk_ids(self, *, project_id: int) -> set[str]:
        query_filter = self._build_filter(project_id=project_id, filters=None)
        chunk_ids: set[str] = set()
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                chunk_id = payload.get("chunk_id")
                if chunk_id:
                    chunk_ids.add(str(chunk_id))
            if offset is None:
                break

        return chunk_ids

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
