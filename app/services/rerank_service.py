from typing import Any

from config import DEFAULT_RERANK_FINAL_TOP_K
from clients.rerank_client import RerankClient


def get_candidate_content(candidate: dict[str, Any]) -> str:
    """
    从候选检索结果中提取用于 Rerank 的完整文本。
    """
    content = candidate.get("content")

    if isinstance(content, str) and content.strip():
        return content

    preview = candidate.get("content_preview")

    if isinstance(preview, str) and preview.strip():
        return preview

    raise ValueError(
        f"候选结果缺少文本内容：{candidate.get('chunk_id')}"
    )


def rerank_candidates(
    query: str,
    candidates: list[dict],
    rerank_client: RerankClient,
    final_top_k: int = DEFAULT_RERANK_FINAL_TOP_K,
) -> list[dict]:
    """
    对第一阶段召回结果重新排序，并保留召回阶段排名与精排分数。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    if final_top_k <= 0:
        raise ValueError("final_top_k 必须大于 0")

    if not candidates:
        return []

    documents = [
        get_candidate_content(candidate)
        for candidate in candidates
    ]
    rerank_scores = rerank_client.score(
        query=query,
        documents=documents,
    )

    if len(rerank_scores) != len(candidates):
        raise ValueError("Rerank 分数数量与候选数量不一致")

    results = []

    for candidate, score in zip(candidates, rerank_scores):
        item = dict(candidate)
        item["retrieval_rank"] = candidate.get("rank")
        item["rerank_score"] = score
        results.append(item)

    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    final_results = results[:final_top_k]

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank

    return final_results
