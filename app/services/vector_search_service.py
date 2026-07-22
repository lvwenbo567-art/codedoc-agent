from typing import Dict, List, Optional

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
from repositories.vector_store import (
    cosine_similarity,
    load_vector_index_bundle,
)


def validate_index_compatibility(
    metadata: Dict,
    query_provider: str,
    query_model: str,
    query_dimension: int,
) -> None:
    """
    校验查询使用的 Embedding 配置是否与索引元数据一致。
    """
    index_version = metadata.get("index_format_version")

    if index_version == "legacy":
        raise ValueError(
            "当前向量索引是旧格式，请重新调用 /index 构建新索引"
        )

    index_provider = metadata.get("embedding_provider")
    index_model = metadata.get("embedding_model")
    index_dimension = metadata.get("dimension")

    if index_provider != query_provider:
        raise ValueError(
            "Embedding Provider 不一致："
            f"索引使用 {index_provider}，查询使用 {query_provider}"
        )

    if index_model != query_model:
        raise ValueError(
            "Embedding 模型不一致："
            f"索引使用 {index_model}，查询使用 {query_model}"
        )

    if index_dimension != query_dimension:
        raise ValueError(
            "Embedding 向量维度不一致："
            f"索引维度 {index_dimension}，查询维度 {query_dimension}"
        )


def rank_vector_records(
    query_embedding: List[float],
    records: List[Dict],
    top_k: int = 5,
    chunk_type: Optional[str] = None,
    include_content: bool = False,
) -> List[Dict]:
    """
    根据余弦相似度对向量记录排序，并返回 Top-K 结果。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    scored_records = []

    for record in records:
        if chunk_type is not None and record["chunk_type"] != chunk_type:
            continue

        embedding = record.get("embedding")

        if not isinstance(embedding, list):
            raise ValueError(
                f"向量记录缺少合法 embedding：{record.get('chunk_id')}"
            )

        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        result = {
            "chunk_id": record["chunk_id"],
            "source_path": record["source_path"],
            "source_name": record["source_name"],
            "source_suffix": record["source_suffix"],
            "chunk_type": record["chunk_type"],
            "chunk_index": record["chunk_index"],
            "content_preview": record["content"][:200],
            "length": record["length"],
            "score": score,
        }

        if include_content:
            result["content"] = record["content"]

        scored_records.append(result)

    scored_records.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = scored_records[:top_k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


def search_vector_records(
    query: str,
    records: List[Dict],
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    chunk_type: Optional[str] = None,
    include_content: bool = False,
) -> List[Dict]:
    """
    兼容旧版本的内存向量检索入口。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    config = EmbeddingConfig(
        provider="mock",
        model_name=model_name,
        mock_dimension=dimension,
        normalize=True,
    )

    embedding_client = EmbeddingClient(config=config)
    query_embedding = embedding_client.embed_text(query)

    return rank_vector_records(
        query_embedding=query_embedding,
        records=records,
        top_k=top_k,
        chunk_type=chunk_type,
        include_content=include_content,
    )


def search_vector_index_from_file(
    query: str,
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
    # Day19/20 compatibility aliases.
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> Dict:
    """
    从向量索引文件中执行完整检索流程。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if model_name is not None and embedding_model is None:
        embedding_model = model_name

    if dimension is not None:
        mock_dimension = dimension

    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    bundle = load_vector_index_bundle(index_path)
    metadata = bundle["metadata"]
    records = bundle["records"]

    # Legacy indexes were created before metadata existed. Keep old behavior for
    # mock-only legacy calls, but force real-provider calls to rebuild.
    if metadata.get("index_format_version") == "legacy":
        if embedding_provider != "mock":
            validate_index_compatibility(
                metadata=metadata,
                query_provider=embedding_provider,
                query_model=embedding_model,
                query_dimension=mock_dimension,
            )

        return {
            "index_path": index_path,
            "query": query,
            "top_k": top_k,
            "model_name": embedding_model,
            "embedding_provider": "mock",
            "embedding_model": embedding_model,
            "dimension": mock_dimension,
            "chunk_type": chunk_type,
            "index_metadata": metadata,
            "result_count": len(
                search_vector_records(
                    query=query,
                    records=records,
                    top_k=top_k,
                    model_name=embedding_model,
                    dimension=mock_dimension,
                    chunk_type=chunk_type,
                    include_content=include_content,
                )
            ),
            "results": search_vector_records(
                query=query,
                records=records,
                top_k=top_k,
                model_name=embedding_model,
                dimension=mock_dimension,
                chunk_type=chunk_type,
                include_content=include_content,
            ),
        }

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

    validate_index_compatibility(
        metadata=metadata,
        query_provider=embedding_provider,
        query_model=embedding_model,
        query_dimension=len(query_embedding),
    )

    results = rank_vector_records(
        query_embedding=query_embedding,
        records=records,
        top_k=top_k,
        chunk_type=chunk_type,
        include_content=include_content,
    )

    return {
        "index_path": index_path,
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
