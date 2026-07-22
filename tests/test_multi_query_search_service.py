from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))


from services.multi_query_search_service import (
    merge_multi_query_results,
    multi_query_hybrid_search,
)


def test_merge_duplicate_chunks():
    query_results = [
        {
            "query": "query one",
            "query_type": "original",
            "search_result": {
                "results": [
                    {"chunk_id": "a", "final_score": 0.8, "content": "A"},
                    {"chunk_id": "b", "final_score": 0.6, "content": "B"},
                ]
            },
        },
        {
            "query": "query two",
            "query_type": "rewrite",
            "search_result": {
                "results": [
                    {"chunk_id": "a", "final_score": 0.9, "content": "A"},
                    {"chunk_id": "c", "final_score": 0.7, "content": "C"},
                ]
            },
        },
    ]

    results = merge_multi_query_results(
        query_results=query_results,
        candidate_top_k=10,
    )

    assert len(results) == 3

    chunk_a = next(item for item in results if item["chunk_id"] == "a")

    assert chunk_a["query_match_count"] == 2
    assert chunk_a["best_retrieval_score"] == 0.9
    assert chunk_a["matched_query_types"] == ["original", "rewrite"]
    assert chunk_a["multi_query_score"] == 0.9500000000000001


def test_merge_supports_string_scores():
    results = merge_multi_query_results(
        query_results=[
            {
                "query": "query",
                "query_type": "original",
                "search_result": {
                    "results": [
                        {
                            "chunk_id": "a",
                            "final_score": "0.8",
                            "content": "A",
                        }
                    ]
                },
            }
        ],
        candidate_top_k=5,
    )

    assert results[0]["best_retrieval_score"] == 0.8


def test_merge_raises_when_chunk_id_missing():
    with pytest.raises(ValueError, match="chunk_id"):
        merge_multi_query_results(
            query_results=[
                {
                    "query": "query",
                    "query_type": "original",
                    "search_result": {
                        "results": [
                            {
                                "final_score": 0.8,
                                "content": "A",
                            }
                        ]
                    },
                }
            ],
            candidate_top_k=5,
        )


def test_multi_query_calls_search_once_per_query():
    call_count = 0

    def fake_search_function(query: str, final_top_k: int, **kwargs) -> dict:
        nonlocal call_count
        call_count += 1
        return {
            "results": [
                {
                    "chunk_id": query,
                    "final_score": 1.0,
                    "content": query,
                }
            ]
        }

    result = multi_query_hybrid_search(
        query_items=[
            {"query": "original", "query_type": "original"},
            {"query": "rewrite", "query_type": "rewrite"},
        ],
        search_function=fake_search_function,
        candidate_top_k=10,
        per_query_top_k=5,
        search_kwargs={},
    )

    assert call_count == 2
    assert result["query_count"] == 2
    assert result["queries"] == result["query_items"]
    assert len(result["query_results"]) == 2
    assert result["candidate_count"] == 2


def test_multi_query_raises_on_empty_query():
    def fake_search_function(query: str, final_top_k: int, **kwargs) -> dict:
        return {"results": []}

    with pytest.raises(ValueError, match="空 Query"):
        multi_query_hybrid_search(
            query_items=[
                {"query": "   ", "query_type": "rewrite"},
            ],
            search_function=fake_search_function,
            candidate_top_k=10,
            per_query_top_k=5,
            search_kwargs={},
        )
