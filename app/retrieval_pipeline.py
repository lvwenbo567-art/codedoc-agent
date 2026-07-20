import logging
import time

from config import (
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
from hybrid_search_service import hybrid_search_from_files
from rerank_client import (
    RerankConfig,
    RerankServiceError,
    get_cached_rerank_client,
)
from rerank_service import rerank_candidates


logger = logging.getLogger(__name__)


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
) -> dict:
    """
    先执行 Hybrid Search 召回候选，再使用 Reranker 对候选重新排序。
    """
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k 必须大于 0")

    if final_top_k <= 0:
        raise ValueError("final_top_k 必须大于 0")

    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k 不能大于 candidate_top_k")

    hybrid_result = hybrid_search_from_files(
        query=query,
        chunks_path=chunks_path,
        index_path=index_path,
        keyword_top_k=candidate_top_k,
        vector_top_k=candidate_top_k,
        final_top_k=candidate_top_k,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_timeout_seconds=embedding_timeout_seconds,
        mock_dimension=mock_dimension,
        chunk_type=chunk_type,
    )
    config = RerankConfig(
        provider=rerank_provider,
        model_name_or_path=rerank_model,
        device=rerank_device,
        batch_size=rerank_batch_size,
        max_length=rerank_max_length,
        local_files_only=rerank_local_files_only,
    )
    rerank_client = get_cached_rerank_client(config=config)
    candidates = hybrid_result["results"]
    rerank_start = time.perf_counter()
    rerank_applied = False
    degraded = False
    degrade_reason = None
    """
    Rerank还没有成功执行
    系统还没有发生降级
    没有降级原因
    """

    try:
        results = rerank_candidates(
            query=query,
            candidates=candidates,
            rerank_client=rerank_client,
            final_top_k=final_top_k,
        )
        rerank_applied = True
        """
    真正的模型服务故障
→ 降级

参数错误、数据错误、代码Bug
→ 直接报错
    """
    except RerankServiceError as exc:
        logger.warning(
            "Rerank 服务异常，降级为 Hybrid Search：%s",
            exc,
        )
        results = [
            dict(item)
            for item in candidates[:final_top_k]
        ]
        degraded = True
        degrade_reason = str(exc)

    rerank_duration_ms = round(
        (time.perf_counter() - rerank_start) * 1000,
        2,
    )

    if rerank_duration_ms > 2000:
        logger.warning(
            "Rerank 耗时较高：%.2f ms",
            rerank_duration_ms,
        )

    return {
        "query": query,
        "retrieval_mode": (
            "hybrid_rerank"
            if rerank_applied
            else "hybrid_fallback"
        ),
        "rerank_applied": rerank_applied,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
        "rerank_duration_ms": rerank_duration_ms,
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
