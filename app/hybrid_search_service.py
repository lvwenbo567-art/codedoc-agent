from typing import Any

from chunk_storage import load_chunks_from_json
from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_VECTOR_INDEX_PATH,
)
from retriever import search_chunks
from search_service import build_search_results
from vector_search_service import search_vector_index_from_file


def normalize_result_scores(
    results: list[dict[str, Any]],
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """
    使用 Min-Max 方法将检索分数归一化到 0～1。
    """
    if not results:
        return []

    scores = [
        float(result[score_key])
        for result in results
    ]
    minimum = min(scores)
    maximum = max(scores)

    normalized_results = []

    for result in results:
        item = dict(result)
        score = float(item[score_key])

        if maximum == minimum:
            normalized_score = 1.0
        else:
            normalized_score = (score - minimum) / (maximum - minimum)

        item["normalized_score"] = normalized_score
        normalized_results.append(item)

    return normalized_results


def validate_hybrid_parameters(
    keyword_weight: float,
    vector_weight: float,
    top_k: int,
) -> None:
    """
    校验混合检索权重和 Top-K 参数是否合法。
    """
    if keyword_weight < 0:
        raise ValueError("keyword_weight 不能小于 0")

    if vector_weight < 0:
        raise ValueError("vector_weight 不能小于 0")

    if keyword_weight + vector_weight <= 0:
        raise ValueError("关键词权重和向量权重不能同时为 0")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")


def merge_hybrid_results(
    keyword_results: list[dict],
    vector_results: list[dict],
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
    top_k: int = 5,
) -> list[dict]:
    """
    合并关键词检索与向量检索结果，按 chunk_id 去重并计算最终融合分数。
    """
    validate_hybrid_parameters(
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        top_k=top_k,
    )

    normalized_keyword_results = normalize_result_scores(keyword_results)
    normalized_vector_results = normalize_result_scores(vector_results)
    merged: dict[str, dict] = {}

    for result in normalized_keyword_results:
        chunk_id = result["chunk_id"]
        merged[chunk_id] = {
            **result,
            "keyword_score": result["score"],
            "keyword_normalized_score": result["normalized_score"],
            "vector_score": None,
            "vector_normalized_score": 0.0,
            "matched_by": ["keyword"],
        }

    for result in normalized_vector_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = {
                **result,
                "keyword_score": None,
                "keyword_normalized_score": 0.0,
                "vector_score": result["score"],
                "vector_normalized_score": result["normalized_score"],
                "matched_by": ["vector"],
            }
            continue

        item = merged[chunk_id]
        item["vector_score"] = result["score"]
        item["vector_normalized_score"] = result["normalized_score"]

        if "vector" not in item["matched_by"]:
            item["matched_by"].append("vector")

    final_results = []

    for item in merged.values():
        final_score = (
            keyword_weight * item["keyword_normalized_score"]
            + vector_weight * item["vector_normalized_score"]
        )
        item["final_score"] = final_score
        item.pop("normalized_score", None)
        final_results.append(item)

    final_results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )
    final_results = final_results[:top_k]

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank

    return final_results


def search_keyword_from_chunks_file(
    query: str,
    chunks_path: str,
    top_k: int,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    从 chunks 文件执行关键词检索，并可按 chunk_type 过滤候选。
    """
    chunks = load_chunks_from_json(chunks_path)

    if chunk_type is not None:
        chunks = [
            chunk
            for chunk in chunks
            if chunk["chunk_type"] == chunk_type
        ]

    retrieved_chunks = search_chunks(
        query=query,
        chunks=chunks,
        top_k=top_k,
    )

    return build_search_results(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )


def hybrid_search_from_files(
    query: str,
    chunks_path: str,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    keyword_top_k: int = 10,
    vector_top_k: int = 10,
    final_top_k: int = 5,
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    chunk_type: str | None = None,
) -> dict:
    """
    执行关键词检索与向量检索，并融合两个通道的结果。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    keyword_results = search_keyword_from_chunks_file(
        query=query,
        chunks_path=chunks_path,
        top_k=keyword_top_k,
        chunk_type=chunk_type,
    )
    vector_result = search_vector_index_from_file(
        query=query,
        index_path=index_path,
        top_k=vector_top_k,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        timeout_seconds=embedding_timeout_seconds,
        mock_dimension=mock_dimension,
        chunk_type=chunk_type,
        include_content=True,
    )
    results = merge_hybrid_results(
        keyword_results=keyword_results,
        vector_results=vector_result["results"],
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        top_k=final_top_k,
    )

    return {
        "query": query,
        "chunks_path": chunks_path,
        "index_path": index_path,
        "keyword_top_k": keyword_top_k,
        "vector_top_k": vector_top_k,
        "final_top_k": final_top_k,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dimension": vector_result.get("dimension"),
        "chunk_type": chunk_type,
        "keyword_result_count": len(keyword_results),
        "vector_result_count": vector_result["result_count"],
        "result_count": len(results),
        "results": results,
    }
