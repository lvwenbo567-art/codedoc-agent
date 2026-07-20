from pathlib import Path
import json
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from chunk_storage import save_chunks_to_json
from index_service import build_vector_index_from_json


client = TestClient(app)


def make_chunk(chunk_id: str, content: str) -> dict:
    """
    构造 API 测试用 chunk。
    """
    return {
        "chunk_id": chunk_id,
        "source_path": f"test_project/{chunk_id}.py",
        "source_name": f"{chunk_id}.py",
        "source_suffix": ".py",
        "chunk_type": "code",
        "chunk_index": 0,
        "content": content,
        "length": len(content),
    }


def prepare_mock_index(tmp_path) -> tuple[Path, Path]:
    """
    准备 /rerank_search API 测试所需 chunks 和索引。
    """
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "vector_index.json"
    chunks = [
        make_chunk(
            "database::chunk_0",
            "SQLite 数据库包含 projects files chunks 表。",
        ),
        make_chunk(
            "embedding::chunk_0",
            "EmbeddingClient 负责生成向量并构建索引。",
        ),
    ]

    save_chunks_to_json(chunks, str(chunks_path))
    build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(index_path),
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        incremental=False,
    )

    return chunks_path, index_path


def test_rerank_search_api_success(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)

    response = client.post(
        "/rerank_search",
        json={
            "query": "EmbeddingClient",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "candidate_top_k": 2,
            "final_top_k": 1,
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "mock_dimension": 32,
            "rerank_provider": "mock",
            "rerank_model": "mock",
        },
    )

    assert response.status_code == 200

    body = response.json()
    data = body["data"]

    assert body["success"] is True
    assert data["retrieval_mode"] == "hybrid_rerank"
    assert data["result_count"] == 1
    assert data["results"][0]["chunk_id"] == "embedding::chunk_0"
    assert data["results"][0]["rank"] == 1
    assert "rerank_score" in data["results"][0]


def test_rerank_search_api_rejects_final_top_k_larger_than_candidate_top_k(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)

    response = client.post(
        "/rerank_search",
        json={
            "query": "EmbeddingClient",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "candidate_top_k": 1,
            "final_top_k": 2,
        },
    )

    assert response.status_code == 400
    assert response.json()["success"] is False


def test_rerank_search_api_missing_file_returns_404(tmp_path):
    response = client.post(
        "/rerank_search",
        json={
            "query": "EmbeddingClient",
            "chunks_path": str(tmp_path / "missing_chunks.json"),
            "index_path": str(tmp_path / "missing_index.json"),
        },
    )

    assert response.status_code == 404
    assert response.json()["success"] is False


def test_rerank_search_api_runtime_error_maps_to_502(monkeypatch):
    def fake_retrieve_with_rerank(**kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        "api_main.retrieve_with_rerank",
        fake_retrieve_with_rerank,
    )

    response = client.post(
        "/rerank_search",
        json={
            "query": "EmbeddingClient",
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "MODEL_SERVICE_ERROR"


def test_rerank_eval_api_success(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)
    eval_path = tmp_path / "rerank_eval.json"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "query": "EmbeddingClient",
                    "expected_chunk_ids": ["embedding::chunk_0"],
                    "question_type": "class_responsibility",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = client.post(
        "/rerank_eval",
        json={
            "eval_path": str(eval_path),
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "candidate_top_k": 2,
            "final_top_k": 2,
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "mock_dimension": 32,
            "rerank_provider": "mock",
            "rerank_model": "mock",
        },
    )

    assert response.status_code == 200

    data = response.json()["data"]

    assert data["hybrid_metrics"]["case_count"] == 1
    assert data["rerank_metrics"]["case_count"] == 1
    assert data["rerank_metrics"]["hit_at_1"] == 1.0
    assert data["rerank_metrics"]["mrr"] == 1.0


def test_rerank_eval_api_missing_eval_file_returns_404(tmp_path):
    response = client.post(
        "/rerank_eval",
        json={
            "eval_path": str(tmp_path / "missing_eval.json"),
        },
    )

    assert response.status_code == 404
    assert response.json()["success"] is False
