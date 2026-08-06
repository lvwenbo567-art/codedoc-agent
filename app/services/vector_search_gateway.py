from __future__ import annotations

from typing import Optional

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_NORMALIZE_EMBEDDING,
    DEFAULT_VECTOR_INDEX_PATH,
)
from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from repositories.vector_store import load_vector_index_bundle
from services.vector_search_service import validate_index_compatibility
from vectorstores.config import VectorStoreConfig
from vectorstores.factory import create_vector_store
from vectorstores.models import VectorSearchFilters


def search_vector_store(
    *,
    query: str,
    project_id: int,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    top_k: int = 5,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: Optional[str] = None,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDING,
    chunk_type: Optional[str] = None,
    include_content: bool = False,
    backend: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "codedoc_chunks_v1",
    qdrant_api_key: str | None = None,
) -> dict:
    """通过统一 VectorStore 执行向量检索，并返回兼容旧接口的结果。"""
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    if project_id <= 0:
        raise ValueError("project_id 必须大于 0")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    config = EmbeddingConfig(
        provider=embedding_provider,
        model_name=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        timeout_seconds=timeout_seconds,
        mock_dimension=mock_dimension,
        normalize=normalize,
    )
    embedding_client = EmbeddingClient(config=config)
    query_embedding = embedding_client.embed_text(query)

    metadata = {}
    if index_path:
        try:
            bundle = load_vector_index_bundle(index_path)
            metadata = bundle["metadata"]
            if metadata.get("index_format_version") != "legacy":
                validate_index_compatibility(
                    metadata=metadata,
                    query_provider=embedding_provider,
                    query_model=embedding_model,
                    query_dimension=len(query_embedding),
                )
        except FileNotFoundError:
            if (backend or VectorStoreConfig.from_env().backend) == "json":
                raise

    store_config = VectorStoreConfig.from_env()
    if backend is not None:
        store_config = store_config.model_copy(update={"backend": backend})
    store_config = store_config.model_copy(
        update={
            "json_index_path": index_path or store_config.json_index_path,
            "project_id": project_id,
            "qdrant_url": qdrant_url,
            "qdrant_collection": qdrant_collection,
            "qdrant_api_key": qdrant_api_key,
        }
    )
    vector_store = create_vector_store(store_config)

    try:
        search_results = vector_store.search(
            project_id=project_id,
            query_vector=query_embedding,
            top_k=top_k,
            filters=VectorSearchFilters(chunk_type=chunk_type),
        )
    finally:
        vector_store.close()

    results = []
    for rank, result in enumerate(search_results, start=1):
        record = result.to_legacy_record(rank=rank)
        if not include_content:
            record.pop("content", None)
        if "content" in result.payload:
            record.setdefault("content_preview", str(result.payload["content"])[:200])
        results.append(record)

    return {
        "backend": store_config.backend,
        "index_path": index_path,
        "project_id": project_id,
        "query": query,
        "top_k": top_k,
        "model_name": embedding_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dimension": len(query_embedding),
        "chunk_type": chunk_type,
        "index_metadata": metadata,
        "result_count": len(results),
        "results": results,
    }
