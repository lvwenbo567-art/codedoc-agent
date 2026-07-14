from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from embedding_client import EmbeddingClient
from vector_store import save_vector_index


client = TestClient(app)


def create_test_index(
    index_path: Path,
    dimension: int = 32,
) -> None:
    embedding_client = EmbeddingClient(
        model_name="test-embedding",
        dimension=dimension,
    )

    content = "def main(): pass"

    records = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": content,
            "length": len(content),
            "embedding": embedding_client.embed_text(content),
        }
    ]

    save_vector_index(
        records=records,
        output_path=str(index_path),
    )


def test_ask_api_success(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    response = client.post(
        "/ask",
        json={
            "query": "def main(): pass",
            "index_path": str(index_path),
            "top_k": 1,
            "embedding_model": "test-embedding",
            "dimension": 32,
            "chat_model": "test-chat",
            "chunk_type": "code",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["query"] == "def main(): pass"
    assert data["data"]["chat_model"] == "test-chat"
    assert data["data"]["chat_provider"] == "mock"
    assert data["data"]["embedding_model"] == "test-embedding"
    assert data["data"]["top_k"] == 1
    assert data["data"]["chunk_type"] == "code"
    assert data["data"]["retrieval_count"] == 1
    assert "[Source 1]" in data["data"]["answer"]
    assert len(data["data"]["citations"]) == 1
    assert data["data"]["citations"][0]["source_path"] == "main.py"
    assert data["data"]["answer_quality"]["is_valid"] is True


def test_ask_api_model_timeout_maps_to_504(monkeypatch):
    def fake_ask_from_vector_index(**kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(
        "api_main.ask_from_vector_index",
        fake_ask_from_vector_index,
    )

    response = client.post(
        "/ask",
        json={
            "query": "main",
        },
    )

    assert response.status_code == 504

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "MODEL_SERVICE_TIMEOUT"


def test_ask_api_model_runtime_error_maps_to_502(monkeypatch):
    def fake_ask_from_vector_index(**kwargs):
        raise RuntimeError("bad gateway")

    monkeypatch.setattr(
        "api_main.ask_from_vector_index",
        fake_ask_from_vector_index,
    )

    response = client.post(
        "/ask",
        json={
            "query": "main",
        },
    )

    assert response.status_code == 502

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "MODEL_SERVICE_ERROR"


def test_ask_api_index_not_exists():
    response = client.post(
        "/ask",
        json={
            "query": "main function?",
            "index_path": "not_exists_vector_index.json",
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


def test_ask_api_empty_query(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    response = client.post(
        "/ask",
        json={
            "query": "   ",
            "index_path": str(index_path),
            "dimension": 32,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"


def test_ask_api_invalid_top_k(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    response = client.post(
        "/ask",
        json={
            "query": "main",
            "index_path": str(index_path),
            "top_k": 0,
            "dimension": 32,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_ask_api_invalid_max_context_chars(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    response = client.post(
        "/ask",
        json={
            "query": "main",
            "index_path": str(index_path),
            "dimension": 32,
            "max_context_chars": 0,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"
