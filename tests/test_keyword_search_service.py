from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.chunk_storage import save_chunks_to_json
from services.keyword_search_service import build_search_results, search_chunks_from_json


def test_build_search_results():
    retrieved_chunks = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
            "score": 8,
        }
    ]

    results = build_search_results(
        query="main",
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1

    result = results[0]

    assert result["query"] == "main"
    assert result["rank"] == 1
    assert result["score"] == 8
    assert result["chunk_id"] == "main.py::chunk_0"
    assert result["source_name"] == "main.py"
    assert result["chunk_type"] == "code"
    assert result["content_preview"] == "def main(): pass"


def test_search_chunks_from_json(tmp_path):
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This is a README file.",
            "length": 22,
        },
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
    ]

    output_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(output_path),
    )

    results = search_chunks_from_json(
        input_path=str(output_path),
        query="main",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["source_name"] == "main.py"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0


def test_search_chunks_from_json_no_match(tmp_path):
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

    output_path = tmp_path / "chunks.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(output_path),
    )

    results = search_chunks_from_json(
        input_path=str(output_path),
        query="notfound",
        top_k=3,
    )

    assert results == []

