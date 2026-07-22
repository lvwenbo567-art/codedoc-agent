from pathlib import Path
import json
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.rerank_eval_service import (
    compare_search_methods,
    evaluate_search_method,
    load_eval_cases,
)


def test_load_eval_cases_success(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "query": "query",
                    "expected_chunk_ids": ["a"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(str(eval_path))

    assert cases[0]["query"] == "query"


def test_load_eval_cases_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_cases(str(tmp_path / "missing.json"))


def test_load_eval_cases_rejects_non_list_root(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        load_eval_cases(str(eval_path))


def test_evaluate_search_method_calculates_metrics():
    eval_cases = [
        {
            "query": "q1",
            "expected_chunk_ids": ["a"],
            "question_type": "type1",
        },
        {
            "query": "q2",
            "expected_chunk_ids": ["z"],
            "question_type": "type2",
        },
    ]

    def search_function(query: str) -> dict:
        if query == "q1":
            return {
                "results": [
                    {"chunk_id": "a"},
                    {"chunk_id": "b"},
                ]
            }

        return {
            "results": [
                {"chunk_id": "x"},
                {"chunk_id": "z"},
            ]
        }

    result = evaluate_search_method(
        eval_cases=eval_cases,
        search_function=search_function,
    )

    assert result["case_count"] == 2
    assert result["hit_at_1"] == 0.5
    assert result["hit_at_3"] == 1.0
    assert result["hit_at_5"] == 1.0
    assert result["mrr"] == 0.75
    assert result["average_latency_ms"] >= 0
    assert result["cases"][0]["question_type"] == "type1"


def test_evaluate_search_method_rejects_empty_eval_cases():
    with pytest.raises(ValueError):
        evaluate_search_method([], lambda query: {"results": []})


def test_evaluate_search_method_rejects_invalid_case():
    with pytest.raises(ValueError):
        evaluate_search_method(
            [
                {
                    "query": "",
                    "expected_chunk_ids": ["a"],
                }
            ],
            lambda query: {"results": []},
        )


def test_compare_search_methods_returns_two_metric_groups():
    eval_cases = [
        {
            "query": "q",
            "expected_chunk_ids": ["a"],
        }
    ]

    def hybrid_search(query: str) -> dict:
        return {"results": [{"chunk_id": "b"}, {"chunk_id": "a"}]}

    def rerank_search(query: str) -> dict:
        return {"results": [{"chunk_id": "a"}, {"chunk_id": "b"}]}

    result = compare_search_methods(
        eval_cases=eval_cases,
        hybrid_search_function=hybrid_search,
        rerank_search_function=rerank_search,
    )

    assert result["hybrid_metrics"]["hit_at_1"] == 0.0
    assert result["rerank_metrics"]["hit_at_1"] == 1.0
