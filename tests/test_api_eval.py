import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app
from services.chunk_storage import save_chunks_to_json


client = TestClient(app)


def test_evaluate_retrieval_api_success(tmp_path):
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

    eval_queries = [
        {
            "query": "main",
            "expected_chunk_ids": [
                "main.py::chunk_0",
            ],
        }
    ]

    eval_path = tmp_path / "eval_queries.json"
    eval_path.write_text(
        json.dumps(eval_queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    response = client.post(
        "/eval",
        json={
            "chunks_path": str(chunks_path),
            "eval_path": str(eval_path),
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["chunks_path"] == str(chunks_path)
    assert data["data"]["top_k"] == 3

    summary = data["data"]["summary"]

    assert summary["query_count"] == 1
    assert summary["avg_hit_rate"] == 1.0
    assert summary["avg_recall"] == 1.0
    assert summary["avg_mrr"] == 1.0


def test_evaluate_retrieval_api_chunks_file_not_exists(tmp_path):
    eval_queries = [
        {
            "query": "main",
            "expected_chunk_ids": [
                "main.py::chunk_0",
            ],
        }
    ]

    eval_path = tmp_path / "eval_queries.json"
    eval_path.write_text(
        json.dumps(eval_queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    response = client.post(
        "/eval",
        json={
            "chunks_path": "not_exists_chunks.json",
            "eval_path": str(eval_path),
            "top_k": 3,
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


def test_evaluate_retrieval_api_eval_file_not_exists(tmp_path):
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
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    response = client.post(
        "/eval",
        json={
            "chunks_path": str(chunks_path),
            "eval_path": "not_exists_eval.json",
            "top_k": 3,
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


def test_evaluate_retrieval_api_invalid_top_k(tmp_path):
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
        }
    ]

    chunks_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    eval_queries = [
        {
            "query": "main",
            "expected_chunk_ids": [
                "main.py::chunk_0",
            ],
        }
    ]

    eval_path = tmp_path / "eval_queries.json"
    eval_path.write_text(
        json.dumps(eval_queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    response = client.post(
        "/eval",
        json={
            "chunks_path": str(chunks_path),
            "eval_path": str(eval_path),
            "top_k": 0,
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"


def test_evaluate_retrieval_api_validation_error():
    response = client.post(
        "/eval",
        json={
            "chunks_path": "outputs/chunks.json",
            "eval_path": "data/eval_queries.json",
            "top_k": "bad",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"
