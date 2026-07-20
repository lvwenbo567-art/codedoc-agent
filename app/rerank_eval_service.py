import json
import time
from collections.abc import Callable
from pathlib import Path

from retrieval_metrics import (
    calculate_hit_at_k,
    calculate_reciprocal_rank,
    extract_chunk_ids,
)


def load_eval_cases(eval_path: str) -> list[dict]:
    """
    读取 Rerank 小型评测集，并校验根对象必须是列表。
    """
    path = Path(eval_path)

    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在：{eval_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("评测文件根对象必须是列表")

    return data


def validate_eval_case(case: dict) -> None:
    """
    校验单条评测用例必须包含 query 和 expected_chunk_ids。
    """
    if not isinstance(case.get("query"), str) or not case["query"].strip():
        raise ValueError("评测用例 query 不能为空")

    expected = case.get("expected_chunk_ids")

    if not isinstance(expected, list) or not expected:
        raise ValueError("expected_chunk_ids 必须是非空列表")


def evaluate_search_method(
    eval_cases: list[dict],
    search_function: Callable[[str], dict],
) -> dict:
    """
    对指定检索函数计算 Hit@1、Hit@3、Hit@5、MRR 和平均延迟。
    """
    if not eval_cases:
        raise ValueError("评测集不能为空")

    case_results = []
    total_hit_1 = 0.0
    total_hit_3 = 0.0
    total_hit_5 = 0.0
    total_rr = 0.0
    total_latency_ms = 0.0

    for case in eval_cases:
        validate_eval_case(case)
        start = time.perf_counter()
        search_result = search_function(case["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        results = search_result["results"]
        expected = case["expected_chunk_ids"]
        hit_1 = calculate_hit_at_k(results, expected, k=1)
        hit_3 = calculate_hit_at_k(results, expected, k=3)
        hit_5 = calculate_hit_at_k(results, expected, k=5)
        reciprocal_rank = calculate_reciprocal_rank(results, expected)

        total_hit_1 += hit_1
        total_hit_3 += hit_3
        total_hit_5 += hit_5
        total_rr += reciprocal_rank
        total_latency_ms += latency_ms
        case_results.append(
            {
                "query": case["query"],
                "question_type": case.get("question_type"),
                "hit_at_1": hit_1,
                "hit_at_3": hit_3,
                "hit_at_5": hit_5,
                "reciprocal_rank": reciprocal_rank,
                "latency_ms": round(latency_ms, 2),
                "retrieved_chunk_ids": extract_chunk_ids(results),
            }
        )

    count = len(eval_cases)

    return {
        "case_count": count,
        "hit_at_1": total_hit_1 / count,
        "hit_at_3": total_hit_3 / count,
        "hit_at_5": total_hit_5 / count,
        "mrr": total_rr / count,
        "average_latency_ms": round(total_latency_ms / count, 2),
        "cases": case_results,
    }


def compare_search_methods(
    eval_cases: list[dict],
    hybrid_search_function: Callable[[str], dict],
    rerank_search_function: Callable[[str], dict],
) -> dict:
    """
    对比 Hybrid Search 和 Hybrid + Rerank 的评测指标。
    """
    return {
        "hybrid_metrics": evaluate_search_method(
            eval_cases=eval_cases,
            search_function=hybrid_search_function,
        ),
        "rerank_metrics": evaluate_search_method(
            eval_cases=eval_cases,
            search_function=rerank_search_function,
        ),
    }
