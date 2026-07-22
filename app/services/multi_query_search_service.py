from __future__ import annotations

from collections.abc import Callable
from typing import Any


HybridSearchFunction = Callable[..., dict]


def _get_retrieval_score(result: dict) -> float:
    """
    从检索结果中取出可比较的召回分数。
    """
    for key in ("final_score", "score", "vector_score", "keyword_score"):
        if key not in result:
            continue

        try:
            return float(result[key])
        except (TypeError, ValueError):
            continue

    return 0.0


def merge_multi_query_results(
    query_results: list[dict],
    candidate_top_k: int,
) -> list[dict]:
    """
    将多个 Query 的 Hybrid Search 结果按 chunk_id 合并。

    同一个 Chunk：
    - 只保留一条记录；
    - 保存命中的 Query；
    - 保存最高召回分数；
    - 命中 Query 越多，给予少量加分。
    """
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k 必须大于 0")

    merged: dict[str, dict] = {}

    for query_result in query_results:
        query_text = query_result["query"]
        query_type = query_result["query_type"]
        search_result = query_result["search_result"]

        for result in search_result.get("results", []):
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                raise ValueError("检索结果缺少 chunk_id")

            score = _get_retrieval_score(result)

            if chunk_id not in merged:
                item = dict(result)
                item["matched_queries"] = [query_text]
                item["matched_query_types"] = [query_type]
                item["best_retrieval_score"] = score
                merged[chunk_id] = item
                continue
            """
            merged = {
    "chunk_001": {
        "chunk_id": "chunk_001",
        "final_score": 0.82,
        "content": "ChatClient 的实现代码",
        "matched_queries": [
            "ChatClient 怎么调用模型？"
        ],
        "matched_query_types": [
            "original"
        ],
        "best_retrieval_score": 0.82,
    }
}
            """
            item = merged[chunk_id]

            if query_text not in item["matched_queries"]:
                item["matched_queries"].append(query_text)

            if query_type not in item["matched_query_types"]:
                item["matched_query_types"].append(query_type)

            item["best_retrieval_score"] = max(
                item["best_retrieval_score"],
                score,
            )

    final_results: list[dict] = []

    for item in merged.values():
        query_match_count = len(item["matched_queries"])
        item["query_match_count"] = query_match_count
        item["multi_query_score"] = (
            item["best_retrieval_score"]
            + 0.05 * (query_match_count - 1)
        )
        final_results.append(item)

    final_results.sort(
        key=lambda item: (
            item["multi_query_score"],
            item["best_retrieval_score"],
            item["query_match_count"],
        ),
        reverse=True,
    )

    final_results = final_results[:candidate_top_k]

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank

    return final_results
    '''
    [
    {
        "chunk_id": "chunk_001",
        "source_path": "src/chat_client.py",
        "source_name": "chat_client.py",
        "chunk_type": "code",
        "chunk_index": 2,
        "content": "class ChatClient:\n    def generate(...): ...",

        # 第一次检索命中时，原有的分数字段
        "vector_score": 0.86,
        "keyword_score": 0.74,
        "final_score": 0.82,

        # 多 Query 合并后新增的字段
        "matched_queries": [
            "ChatClient 怎么调用模型？",
            "ChatClient 模型请求流程",
        ],
        "matched_query_types": [
            "original",
            "rewrite",
        ],
        "best_retrieval_score": 0.88,
        "query_match_count": 2,
        "multi_query_score": 0.93,
        "rank": 1,
    },
    {
        "chunk_id": "chunk_003",
        "source_path": "src/config.py",
        "source_name": "config.py",
        "chunk_type": "code",
        "chunk_index": 1,
        "content": "DEFAULT_CHAT_BASE_URL = ...",

        "vector_score": 0.80,
        "keyword_score": 0.76,
        "final_score": 0.81,

        "matched_queries": [
            "ChatClient 模型请求流程",
        ],
        "matched_query_types": [
            "rewrite",
        ],
        "best_retrieval_score": 0.81,
        "query_match_count": 1,
        "multi_query_score": 0.81,
        "rank": 2,
    },
    {
        "chunk_id": "chunk_002",
        "source_path": "docs/chat.md",
        "source_name": "chat.md",
        "chunk_type": "document",
        "chunk_index": 0,
        "content": "ChatClient 用于调用兼容 OpenAI 接口的模型……",

        "vector_score": 0.77,
        "keyword_score": 0.71,
        "final_score": 0.79,

        "matched_queries": [
            "ChatClient 怎么调用模型？",
        ],
        "matched_query_types": [
            "original",
        ],
        "best_retrieval_score": 0.79,
        "query_match_count": 1,
        "multi_query_score": 0.79,
        "rank": 3,
    },
]
    '''

def multi_query_hybrid_search(
    query_items: list[dict],
    search_function: HybridSearchFunction,
    candidate_top_k: int,
    per_query_top_k: int,
    search_kwargs: dict[str, Any],
) -> dict:
    """
    对多个 Query 分别执行第一阶段 Hybrid Search，合并后返回候选。

    本函数不执行 Rerank。
    """
    if not query_items:
        raise ValueError("query_items 不能为空")

    if per_query_top_k <= 0:
        raise ValueError("per_query_top_k 必须大于 0")

    query_results: list[dict] = []

    for query_item in query_items:
        query_text = query_item.get("query", "").strip()
        query_type = query_item.get("query_type", "rewrite")

        if not query_text:
            raise ValueError("query_items 中存在空 Query")

        search_result = search_function(
            query=query_text,
            final_top_k=per_query_top_k,
            **search_kwargs,
        )
        query_results.append(
            {
                "query": query_text,
                "query_type": query_type,
                "search_result": search_result,
            }
        )

    merged_results = merge_multi_query_results(
        query_results=query_results,
        candidate_top_k=candidate_top_k,
    )

    return {
        "query_count": len(query_items),
        "queries": query_items,
        "query_items": query_items,
        "query_results": query_results,
        "candidate_count": len(merged_results),
        "results": merged_results,
    }
