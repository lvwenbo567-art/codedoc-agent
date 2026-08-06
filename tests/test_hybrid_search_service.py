from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.chunk_storage import save_chunks_to_json
from services.hybrid_search_service import (
    hybrid_search_from_files,
    merge_hybrid_results,
    normalize_result_scores,
    validate_hybrid_parameters,
)
from services.index_service import build_vector_index_from_json


def make_chunk(
    chunk_id: str,
    content: str,
    chunk_type: str = "code",
) -> dict:
    """
    构造测试用 chunk，减少每个测试里的重复字段。
    """
    return {
        "chunk_id": chunk_id,
        "source_path": f"test_project/{chunk_id}.py",
        "source_name": f"{chunk_id}.py",
        "source_suffix": ".py",
        "chunk_type": chunk_type,
        "chunk_index": 0,
        "content": content,
        "length": len(content),
    }


def test_normalize_result_scores_uses_min_max():
    results = normalize_result_scores(
        [
            {"chunk_id": "a", "score": 2},
            {"chunk_id": "b", "score": 4},
        ]
    )

    assert results[0]["normalized_score"] == 0
    assert results[1]["normalized_score"] == 1


def test_normalize_result_scores_handles_equal_scores():
    results = normalize_result_scores(
        [
            {"chunk_id": "a", "score": 3},
            {"chunk_id": "b", "score": 3},
        ]
    )

    assert [item["normalized_score"] for item in results] == [1.0, 1.0]


def test_merge_hybrid_results_deduplicates_same_chunk():
    results = merge_hybrid_results(
        keyword_results=[
            {"chunk_id": "a", "score": 2, "content": "keyword hit"}
        ],
        vector_results=[
            {"chunk_id": "a", "score": 0.8, "content": "vector hit"}
        ],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"
    assert results[0]["matched_by"] == ["keyword", "vector"]
    assert results[0]["keyword_score"] == 2
    assert results[0]["vector_score"] == 0.8


def test_merge_hybrid_results_keeps_single_channel_hits():
    results = merge_hybrid_results(
        keyword_results=[
            {"chunk_id": "keyword_only", "score": 1, "content": "keyword"}
        ],
        vector_results=[
            {"chunk_id": "vector_only", "score": 1, "content": "vector"}
        ],
        top_k=5,
    )

    chunk_ids = {item["chunk_id"] for item in results}

    assert chunk_ids == {"keyword_only", "vector_only"}


def test_merge_hybrid_results_respects_top_k():
    results = merge_hybrid_results(
        keyword_results=[
            {"chunk_id": "a", "score": 3},
            {"chunk_id": "b", "score": 2},
            {"chunk_id": "c", "score": 1},
        ],
        vector_results=[],
        top_k=2,
    )

    assert len(results) == 2
    assert [item["rank"] for item in results] == [1, 2]


@pytest.mark.parametrize(
    "keyword_weight, vector_weight, top_k",
    [
        (-0.1, 0.6, 5),
        (0.4, -0.1, 5),
        (0, 0, 5),
        (0.4, 0.6, 0),
    ],
)
def test_validate_hybrid_parameters_rejects_invalid_values(
    keyword_weight,
    vector_weight,
    top_k,
):
    with pytest.raises(ValueError):
        validate_hybrid_parameters(
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            top_k=top_k,
        )


def test_hybrid_search_from_files_with_mock_embedding(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "vector_index.json"
    chunks = [
        make_chunk(
            "api::chunk_0",
            "FastAPI exposes a search endpoint for querying chunks.",
        ),
        make_chunk(
            "database::chunk_0",
            "SQLite stores scanned projects files and chunks.",
        ),
        make_chunk(
            "readme::chunk_0",
            "The project can be started with uvicorn api_main:app.",
            "document",
        ),
    ]

    save_chunks_to_json(chunks, str(chunks_path))
    build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(index_path),
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        batch_size=2,
        incremental=False,
    )

    result = hybrid_search_from_files(
        query="how to search chunks",
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        keyword_top_k=2,
        vector_top_k=2,
        final_top_k=2,
    )

    assert result["result_count"] == 2
    assert result["keyword_result_count"] <= 2
    assert result["vector_result_count"] <= 2
    assert result["results"][0]["rank"] == 1
    assert "final_score" in result["results"][0]
    assert any(
        item.get("keyword_score_type") == "bm25"
        for item in result["results"]
        if "keyword" in item["matched_by"]
    )
