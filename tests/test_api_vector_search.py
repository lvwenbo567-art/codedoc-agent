from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from clients.embedding_client import EmbeddingClient
from repositories.vector_store import save_vector_index


client = TestClient(app)


def create_test_index(
    index_path: Path,
    dimension: int = 32,
) -> None:
    embedding_client = EmbeddingClient(
        model_name="test-model",
        dimension=dimension,
    )

    records = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
            "embedding": embedding_client.embed_text("def main(): pass"),
        },
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This project reads markdown files.",
            "length": 34,
            "embedding": embedding_client.embed_text(
                "This project reads markdown files."
            ),
        },
    ]

    save_vector_index(
        records=records,
        output_path=str(index_path),
    )


def test_vector_search_api_success(tmp_path):
    index_path = tmp_path / "vector_index.json"

    create_test_index(
        index_path=index_path,
        dimension=32,
    )

    response = client.post(
        "/vector_search",
        json={
            "index_path": str(index_path),
            "query": "def main(): pass",
            "top_k": 3,
            "model_name": "test-model",
            "dimension": 32,
            "chunk_type": "code",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["index_path"] == str(index_path)
    assert data["data"]["query"] == "def main(): pass"
    assert data["data"]["top_k"] == 3
    assert data["data"]["chunk_type"] == "code"
    assert data["data"]["result_count"] == 1
    assert data["data"]["results"][0]["source_name"] == "main.py"
    assert data["data"]["results"][0]["rank"] == 1
    assert data["data"]["results"][0]["score"] > 0.99


def test_vector_search_api_top_k(tmp_path):
    index_path = tmp_path / "vector_index.json"

    create_test_index(index_path)

    response = client.post(
        "/vector_search",
        json={
            "index_path": str(index_path),
            "query": "main",
            "top_k": 1,
            "model_name": "test-model",
            "dimension": 32,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["result_count"] == 1
    assert len(data["data"]["results"]) == 1
    assert data["data"]["results"][0]["rank"] == 1


def test_vector_search_api_file_not_exists():
    response = client.post(
        "/vector_search",
        json={
            "index_path": "not_exists_vector_index.json",
            "query": "main",
            "top_k": 3,
            "dimension": 32,
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


def test_vector_search_api_empty_query(tmp_path):
    index_path = tmp_path / "vector_index.json"

    create_test_index(index_path)

    response = client.post(
        "/vector_search",
        json={
            "index_path": str(index_path),
            "query": "   ",
            "top_k": 3,
            "dimension": 32,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"


def test_vector_search_api_invalid_top_k(tmp_path):
    index_path = tmp_path / "vector_index.json"

    create_test_index(index_path)

    response = client.post(
        "/vector_search",
        json={
            "index_path": str(index_path),
            "query": "main",
            "top_k": 0,
            "dimension": 32,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"


def test_vector_search_api_invalid_dimension(tmp_path):
    index_path = tmp_path / "vector_index.json"

    create_test_index(index_path)

    response = client.post(
        "/vector_search",
        json={
            "index_path": str(index_path),
            "query": "main",
            "top_k": 3,
            "dimension": 0,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"
