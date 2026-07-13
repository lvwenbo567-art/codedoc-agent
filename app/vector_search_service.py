from typing import Dict, List, Optional

from config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_INDEX_PATH,
)
from embedding_client import EmbeddingClient
from vector_store import cosine_similarity, load_vector_index


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
    在向量记录中检索与 query 最相似的 Top-K chunks。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    embedding_client = EmbeddingClient(
        model_name=model_name,
        dimension=dimension,
    )

    query_embedding = embedding_client.embed_text(query)

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


def search_vector_index_from_file(
    query: str,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    chunk_type: Optional[str] = None,
    include_content: bool = False,
) -> Dict:
    records = load_vector_index(index_path)

    results = search_vector_records(
        query=query,
        records=records,
        top_k=top_k,
        model_name=model_name,
        dimension=dimension,
        chunk_type=chunk_type,
        include_content=include_content,
    )

    return {
        "index_path": index_path,
        "query": query,
        "top_k": top_k,
        "model_name": model_name,
        "dimension": dimension,
        "chunk_type": chunk_type,
        "result_count": len(results),
        "results": results,
    }