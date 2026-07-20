from pathlib import Path
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
    准备临时 chunks 文件和 mock 向量索引，避免依赖真实模型。
    """
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "vector_index.json"
    chunks = [
        make_chunk(
            "api::chunk_0",
            "FastAPI hybrid search combines keyword and vector retrieval.",
        ),
        make_chunk(
            "rag::chunk_0",
            "RAG answers user questions with retrieved context.",
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


def test_hybrid_search_api_success(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)

    response = client.post(
        "/hybrid_search",
        json={
            "query": "hybrid search",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "mock_dimension": 32,
            "keyword_top_k": 2,
            "vector_top_k": 2,
            "final_top_k": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()
    data = body["data"]

    assert body["success"] is True
    assert data["result_count"] > 0
    assert data["results"][0]["rank"] == 1
    assert "final_score" in data["results"][0]
    assert "matched_by" in data["results"][0]
    assert "keyword_score" in data["results"][0]
    assert "vector_score" in data["results"][0]


def test_hybrid_search_api_rejects_zero_total_weight(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)

    response = client.post(
        "/hybrid_search",
        json={
            "query": "hybrid search",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "mock_dimension": 32,
            "keyword_weight": 0,
            "vector_weight": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["success"] is False


def test_hybrid_search_api_returns_404_for_missing_chunks_file(tmp_path):
    index_path = tmp_path / "missing_index.json"

    response = client.post(
        "/hybrid_search",
        json={
            "query": "hybrid search",
            "chunks_path": str(tmp_path / "missing_chunks.json"),
            "index_path": str(index_path),
            "embedding_provider": "mock",
            "embedding_model": "test-model",
        },
    )

    assert response.status_code == 404
    assert response.json()["success"] is False


def test_hybrid_search_api_validates_final_top_k(tmp_path):
    chunks_path, index_path = prepare_mock_index(tmp_path)

    response = client.post(
        "/hybrid_search",
        json={
            "query": "hybrid search",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "final_top_k": 0,
        },
    )

    assert response.status_code == 422
    assert response.json()["success"] is False
