from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from services.chunk_storage import save_chunks_to_json
from repositories.vector_store import load_vector_index


client = TestClient(app)


def build_test_chunks():
    return [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
        }
    ]


def test_build_vector_index_api_success(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"

    save_chunks_to_json(
        chunks=build_test_chunks(),
        output_path=str(chunks_path),
    )

    response = client.post(
        "/index",
        json={
            "chunks_path": str(chunks_path),
            "output_path": str(output_path),
            "model_name": "test-model",
            "dimension": 32,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["chunks_path"] == str(chunks_path)
    assert data["data"]["output_path"] == str(output_path)
    assert data["data"]["model_name"] == "test-model"
    assert data["data"]["dimension"] == 32
    assert data["data"]["chunk_count"] == 1
    assert data["data"]["vector_count"] == 1

    records = load_vector_index(str(output_path))

    assert len(records) == 1
    assert records[0]["chunk_id"] == "main.py::chunk_0"
    assert len(records[0]["embedding"]) == 32


def test_build_vector_index_api_file_not_exists(tmp_path):
    response = client.post(
        "/index",
        json={
            "chunks_path": "not_exists_chunks.json",
            "output_path": str(tmp_path / "vector_index.json"),
            "model_name": "test-model",
            "dimension": 32,
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


def test_build_vector_index_api_invalid_dimension(tmp_path):
    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=build_test_chunks(),
        output_path=str(chunks_path),
    )

    response = client.post(
        "/index",
        json={
            "chunks_path": str(chunks_path),
            "output_path": str(tmp_path / "vector_index.json"),
            "dimension": 0,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"
