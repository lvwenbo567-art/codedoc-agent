from typing import Dict, List, Set

from services.keyword_search_service import search_chunks_from_json

def calculate_hit_rate(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
) -> float:
    """
    计算单个 query 的 HitRate。

    只要检索结果中命中至少一个 expected chunk，就算命中。
    """
    expected_set = set(expected_chunk_ids)

    for chunk_id in retrieved_chunk_ids:
        if chunk_id in expected_set:
            return 1.0

    return 0.0

def calculate_recall(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
) -> float:
    """
    计算单个 query 的 Recall。

    Recall = 命中的正确 chunk 数量 / 所有正确 chunk 数量
    """
    if not expected_chunk_ids:
        return 0.0

    retrieved_set = set(retrieved_chunk_ids)
    expected_set = set(expected_chunk_ids)

    hit_count = len(retrieved_set & expected_set)

    return hit_count / len(expected_set)

def calculate_mrr(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
) -> float:
    """
    计算单个 query 的 Reciprocal Rank。

    第一个正确结果排第 1，得 1/1。
    第一个正确结果排第 2，得 1/2。
    没有命中，得 0。
    """
    expected_set = set(expected_chunk_ids)

    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected_set:
            return 1.0 / index

    return 0.0


def evaluate_single_query(
    query: str,
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
) -> Dict:
    """
    评估单个 query 的检索效果。
    """
    hit_rate = calculate_hit_rate(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    recall = calculate_recall(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    mrr = calculate_mrr(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    return {
        "query": query,
        "hit_rate": hit_rate,
        "recall": recall,
        "mrr": mrr,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "expected_chunk_ids": expected_chunk_ids,
    }

def evaluate_queries(
    eval_items: List[Dict],
) -> Dict:
    """
    汇总多个 query 的平均评估结果。
    """
    if not eval_items:
        return {
            "query_count": 0,
            "avg_hit_rate": 0.0,
            "avg_recall": 0.0,
            "avg_mrr": 0.0,
            "items": [],
        }

    query_count = len(eval_items)

    avg_hit_rate = sum(item["hit_rate"] for item in eval_items) / query_count
    avg_recall = sum(item["recall"] for item in eval_items) / query_count
    avg_mrr = sum(item["mrr"] for item in eval_items) / query_count

    return {
        "query_count": query_count,
        "avg_hit_rate": avg_hit_rate,
        "avg_recall": avg_recall,
        "avg_mrr": avg_mrr,
        "items": eval_items,
    }

def evaluate_from_json(
    chunks_path: str,
    eval_queries: List[Dict],
    top_k: int = 5,
) -> Dict:
    """
    从 chunks.json 中执行检索，并根据人工标注的 expected_chunk_ids 评估检索效果。
    """
    eval_items = []

    for item in eval_queries:
        query = item["query"]
        expected_chunk_ids = item["expected_chunk_ids"]

        results = search_chunks_from_json(
            input_path=chunks_path,
            query=query,
            top_k=top_k,
        )

        retrieved_chunk_ids = [result["chunk_id"] for result in results]

        eval_item = evaluate_single_query(
            query=query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            expected_chunk_ids=expected_chunk_ids,
        )

        eval_items.append(eval_item)

    summary = evaluate_queries(eval_items)

    return {
        "chunks_path": chunks_path,
        "top_k": top_k,
        "summary": summary,
    }
