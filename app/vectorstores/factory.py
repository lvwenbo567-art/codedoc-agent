from __future__ import annotations

from vectorstores.base import VectorStore
from vectorstores.config import VectorStoreConfig
from vectorstores.json_vector_store import JsonVectorStore
from vectorstores.qdrant_vector_store import QdrantVectorStore
from vectorstores.async_base import AsyncVectorStore
from vectorstores.async_json_vector_store import AsyncJsonVectorStore
from vectorstores.async_qdrant_vector_store import AsyncQdrantVectorStore


def create_vector_store(config: VectorStoreConfig | None = None) -> VectorStore:
    """根据配置创建 VectorStore，后端判断只允许出现在 Factory。"""
    config = config or VectorStoreConfig.from_env()

    if config.backend == "json":
        return JsonVectorStore(
            index_path=config.json_index_path,
            project_id=config.project_id,
        )

    return QdrantVectorStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection,
        prefer_grpc=config.qdrant_prefer_grpc,
        timeout_seconds=config.qdrant_timeout_seconds,
    )


def create_async_vector_store(
    config: VectorStoreConfig | None = None,
) -> AsyncVectorStore:
    """根据 Day41 配置创建非阻塞 VectorStore。"""
    config = config or VectorStoreConfig.from_env()
    if config.backend == "json":
        return AsyncJsonVectorStore(
            store=JsonVectorStore(
                index_path=config.json_index_path,
                project_id=config.project_id,
            )
        )
    return AsyncQdrantVectorStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection,
        prefer_grpc=config.qdrant_prefer_grpc,
        timeout_seconds=config.qdrant_timeout_seconds,
    )
