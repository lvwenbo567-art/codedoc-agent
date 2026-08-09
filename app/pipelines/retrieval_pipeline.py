from __future__ import annotations

import logging
import time
from typing import Literal

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_FINAL_TOP_K,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
    DEFAULT_VECTOR_INDEX_PATH,
)
from services.hybrid_search_service import hybrid_search_from_files
from services.multi_query_search_service import multi_query_hybrid_search
from services.query_rewrite_service import QueryRewriteService
from clients.rerank_client import (
    RerankConfig,
    RerankServiceError,
    get_cached_rerank_client,
)
from services.rerank_service import rerank_candidates


logger = logging.getLogger(__name__)

QueryStrategy = Literal["original", "rewrite", "multi_query"]


def _build_unique_query_items(
    original_query: str,
    rewritten_queries: list[str],
) -> list[dict]:
    """
    构建去重后的 Multi-Query 检索项。
    """
    items = [
        {
            "query": original_query,
            "query_type": "original",
        }
    ]
    seen = {original_query}

    for rewritten_query in rewritten_queries:
        value = rewritten_query.strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        items.append(
            {
                "query": value,
                "query_type": "rewrite",
            }
        )

    return items


def _validate_top_k(candidate_top_k: int, final_top_k: int) -> None:
    """
    校验召回候选数量和最终返回数量。
    """
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k 必须大于 0")

    if final_top_k <= 0:
        raise ValueError("final_top_k 必须大于 0")

    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k 不能大于 candidate_top_k")


def _add_single_query_match_metadata(
    candidates: list[dict],
    query: str,
    query_type: str,
) -> None:
    """
    为非 Multi-Query 的召回候选补充和 Multi-Query 一致的命中元数据。
    """
    for candidate in candidates:
        candidate["matched_queries"] = [query]
        candidate["matched_query_types"] = [query_type]
        candidate["query_match_count"] = 1


def _build_query_items(
    query: str,
    query_strategy: QueryStrategy,
    rewrite_count: int,
    query_rewrite_service: QueryRewriteService | None,
    query_rewrite_provider: str,
    query_rewrite_model: str,
    query_rewrite_base_url: str,
    query_rewrite_api_key: str,
    query_rewrite_timeout_seconds: float,
) -> tuple[list[dict], dict | None]:
    """
    根据策略生成实际用于检索的 query 列表。
    """
    if query_strategy == "original":
        return [{"query": query, "query_type": "original"}], None

    service = query_rewrite_service or QueryRewriteService.from_config(
        provider=query_rewrite_provider,
        model_name=query_rewrite_model,
        base_url=query_rewrite_base_url,
        api_key=query_rewrite_api_key,
        timeout_seconds=query_rewrite_timeout_seconds,
    )
    rewrite_result = service.rewrite(
        query=query,
        rewrite_count=rewrite_count,
    )
    rewrite_queries = rewrite_result.get("rewritten_queries", [])

    if query_strategy == "rewrite":
        if rewrite_result.get("fallback_used"):
            return [{"query": query, "query_type": "original"}], rewrite_result

        first_query = rewrite_queries[0] if rewrite_queries else query
        return [{"query": first_query, "query_type": "rewrite"}], rewrite_result

    if query_strategy == "multi_query":
        query_items = _build_unique_query_items(
            original_query=query,
            rewritten_queries=rewrite_queries,
        )
        return query_items, rewrite_result

    raise ValueError(f"不支持的 query_strategy：{query_strategy}")


def retrieve_with_rerank(
    query: str,
    chunks_path: str,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    candidate_top_k: int = DEFAULT_RERANK_CANDIDATE_TOP_K,
    final_top_k: int = DEFAULT_RERANK_FINAL_TOP_K,
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    rerank_provider: str = DEFAULT_RERANK_PROVIDER,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    rerank_device: str = DEFAULT_RERANK_DEVICE,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    rerank_max_length: int = DEFAULT_RERANK_MAX_LENGTH,
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY,
    chunk_type: str | None = None,
    query_strategy: QueryStrategy = "original",
    rewrite_count: int = 2,
    query_rewrite_provider: str = DEFAULT_CHAT_PROVIDER,
    query_rewrite_model: str = DEFAULT_CHAT_MODEL,
    query_rewrite_base_url: str = DEFAULT_CHAT_BASE_URL,
    query_rewrite_api_key: str = DEFAULT_CHAT_API_KEY,
    query_rewrite_timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
    query_rewrite_service: QueryRewriteService | None = None,
    rerank_client_override=None,
) -> dict:
    """
    先执行 Hybrid Search 召回候选，再使用 Reranker 对候选重新排序。

    Day29 新增 query_strategy：
    - original：原始 query 召回
    - rewrite：改写 query 召回
    - multi_query：原始 query + 多条改写 query 多路召回，合并后只 rerank 一次
    """
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")

    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    _validate_top_k(candidate_top_k=candidate_top_k, final_top_k=final_top_k)

    if rewrite_count <= 0:
        raise ValueError("rewrite_count 必须大于 0")

    if query_strategy not in {"original", "rewrite", "multi_query"}:
        raise ValueError(f"不支持的 query_strategy：{query_strategy}")

    pipeline_start = time.perf_counter()

    query_items, rewrite_result = _build_query_items(
        query=query,
        query_strategy=query_strategy,
        rewrite_count=rewrite_count,
        query_rewrite_service=query_rewrite_service,
        query_rewrite_provider=query_rewrite_provider,
        query_rewrite_model=query_rewrite_model,
        query_rewrite_base_url=query_rewrite_base_url,
        query_rewrite_api_key=query_rewrite_api_key,
        query_rewrite_timeout_seconds=query_rewrite_timeout_seconds,
    )

    search_kwargs = {
        "chunks_path": chunks_path,
        "index_path": index_path,
        "keyword_top_k": candidate_top_k,
        "vector_top_k": candidate_top_k,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
        "embedding_timeout_seconds": embedding_timeout_seconds,
        "mock_dimension": mock_dimension,
        "chunk_type": chunk_type,
    }

    retrieval_start = time.perf_counter()

    if query_strategy == "multi_query":
        multi_query_result = multi_query_hybrid_search(
            query_items=query_items,
            search_function=hybrid_search_from_files,
            candidate_top_k=candidate_top_k,
            per_query_top_k=candidate_top_k,
            search_kwargs=search_kwargs,
        )
        candidates = multi_query_result["results"]
        hybrid_result = {
            "results": candidates,
            "dimension": None,
            "query_results": multi_query_result["query_results"],
        }
    else:
        hybrid_result = hybrid_search_from_files(
            query=query_items[0]["query"],
            final_top_k=candidate_top_k,
            **search_kwargs,
        )
        candidates = hybrid_result["results"]
        _add_single_query_match_metadata(
            candidates=candidates,
            query=query_items[0]["query"],
            query_type=query_items[0]["query_type"],
        )

    retrieval_duration_ms = round(
        (time.perf_counter() - retrieval_start) * 1000,
        2,
    )

    config = RerankConfig(
        provider=rerank_provider,
        model_name_or_path=rerank_model,
        device=rerank_device,
        batch_size=rerank_batch_size,
        max_length=rerank_max_length,
        local_files_only=rerank_local_files_only,
    )
    rerank_client = rerank_client_override or get_cached_rerank_client(config=config)

    rerank_start = time.perf_counter()
    rerank_applied = False
    degraded = False
    degrade_reason = None

    try:
        results = rerank_candidates(
            query=query,
            candidates=candidates,
            rerank_client=rerank_client,
            final_top_k=final_top_k,
        )
        rerank_applied = True

    except RerankServiceError as exc:
        logger.warning("Rerank 服务异常，降级为 Hybrid Search：%s", exc)
        results = [dict(item) for item in candidates[:final_top_k]]
        degraded = True
        degrade_reason = str(exc)

    rerank_duration_ms = round((time.perf_counter() - rerank_start) * 1000, 2)

    if rerank_duration_ms > 2000:
        logger.warning("Rerank 耗时较高：%.2f ms", rerank_duration_ms)

    total_duration_ms = round(
        (time.perf_counter() - pipeline_start) * 1000,
        2,
    )

    return {
        "query": query,
        "query_strategy": query_strategy,
        "query_items": query_items,
        "rewrite_result": rewrite_result,
        "retrieval_mode": "hybrid_rerank" if rerank_applied else "hybrid_fallback",
        "rerank_applied": rerank_applied,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
        "retrieval_duration_ms": retrieval_duration_ms,
        "rerank_duration_ms": rerank_duration_ms,
        "total_duration_ms": total_duration_ms,
        "chunks_path": chunks_path,
        "index_path": index_path,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "candidate_count": len(candidates),
        "result_count": len(results),
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dimension": hybrid_result.get("dimension"),
        "chunk_type": chunk_type,
        "rerank_provider": rerank_provider,
        "rerank_model": rerank_model,
        "results": results,
    }
