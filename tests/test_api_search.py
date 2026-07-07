from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from chunk_storage import save_chunks_to_json


client = TestClient(app)


def test_search_chunks_api_success(tmp_path):
    chunks = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
        },
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This is a test project.",
            "length": 23,
        },
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    response = client.post(
        "/search",
        json={
            "chunks_path": str(chunks_path),
            "query": "main",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["query"] == "main"
    assert data["data"]["top_k"] == 3
    assert data["data"]["result_count"] == 1

    result = data["data"]["results"][0]

    assert result["source_name"] == "main.py"
    assert result["chunk_type"] == "code"
    assert result["score"] > 0


def test_search_chunks_api_no_match(tmp_path):
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello world",
            "length": 11,
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    response = client.post(
        "/search",
        json={
            "chunks_path": str(chunks_path),
            "query": "notfound",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["result_count"] == 0
    assert data["data"]["results"] == []


def test_search_chunks_api_file_not_exists():
    response = client.post(
        "/search",
        json={
            "chunks_path": "not_exists_chunks.json",
            "query": "main",
            "top_k": 3,
        },
    )

    assert response.status_code == 404


def test_search_chunks_api_empty_query(tmp_path):
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello world",
            "length": 11,
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    response = client.post(
        "/search",
        json={
            "chunks_path": str(chunks_path),
            "query": "   ",
            "top_k": 3,
        },
    )

    assert response.status_code == 400


def test_search_chunks_api_invalid_top_k(tmp_path):
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello world",
            "length": 11,
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    response = client.post(
        "/search",
        json={
            "chunks_path": str(chunks_path),
            "query": "hello",
            "top_k": 0,
        },
    )

    assert response.status_code == 400