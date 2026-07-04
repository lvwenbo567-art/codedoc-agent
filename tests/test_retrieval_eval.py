from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from retrieval_eval import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_recall,
    evaluate_queries,
    evaluate_single_query,
)


def test_calculate_hit_rate_hit():
    retrieved_chunk_ids = ["a", "b", "c"]
    expected_chunk_ids = ["b"]

    hit_rate = calculate_hit_rate(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert hit_rate == 1.0


def test_calculate_hit_rate_miss():
    retrieved_chunk_ids = ["a", "c"]
    expected_chunk_ids = ["b"]

    hit_rate = calculate_hit_rate(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert hit_rate == 0.0


def test_calculate_recall():
    retrieved_chunk_ids = ["a", "b", "c"]
    expected_chunk_ids = ["b", "c", "d"]

    recall = calculate_recall(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert recall == 2 / 3


def test_calculate_recall_empty_expected():
    recall = calculate_recall(
        retrieved_chunk_ids=["a", "b"],
        expected_chunk_ids=[],
    )

    assert recall == 0.0


def test_calculate_mrr_rank_1():
    retrieved_chunk_ids = ["a", "b", "c"]
    expected_chunk_ids = ["a"]

    mrr = calculate_mrr(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert mrr == 1.0


def test_calculate_mrr_rank_2():
    retrieved_chunk_ids = ["a", "b", "c"]
    expected_chunk_ids = ["b"]

    mrr = calculate_mrr(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert mrr == 0.5


def test_calculate_mrr_miss():
    retrieved_chunk_ids = ["a", "c"]
    expected_chunk_ids = ["b"]

    mrr = calculate_mrr(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    assert mrr == 0.0


def test_evaluate_single_query():
    result = evaluate_single_query(
        query="main",
        retrieved_chunk_ids=["a", "b", "c"],
        expected_chunk_ids=["b", "d"],
    )

    assert result["query"] == "main"
    assert result["hit_rate"] == 1.0
    assert result["recall"] == 0.5
    assert result["mrr"] == 0.5


def test_evaluate_queries():
    eval_items = [
        {
            "query": "q1",
            "hit_rate": 1.0,
            "recall": 1.0,
            "mrr": 1.0,
        },
        {
            "query": "q2",
            "hit_rate": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
        },
    ]

    summary = evaluate_queries(eval_items)

    assert summary["query_count"] == 2
    assert summary["avg_hit_rate"] == 0.5
    assert summary["avg_recall"] == 0.5
    assert summary["avg_mrr"] == 0.5


def test_evaluate_queries_empty():
    summary = evaluate_queries([])

    assert summary["query_count"] == 0
    assert summary["avg_hit_rate"] == 0.0
    assert summary["avg_recall"] == 0.0
    assert summary["avg_mrr"] == 0.0
    assert summary["items"] == []