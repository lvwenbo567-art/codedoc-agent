from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from embedding_client import EmbeddingClient
from vector_search_service import (
    search_vector_index_from_file,
    search_vector_records,
)
from vector_store import save_vector_index


def build_vector_records(dimension: int = 32):
    client = EmbeddingClient(
        model_name="test-model",
        dimension=dimension,
    )

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
            "embedding": client.embed_text("def main(): pass"),
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
            "embedding": client.embed_text("This project reads markdown files."),
        },
    ]


def test_vector_search_records_returns_best_match():
    records = build_vector_records(dimension=32)

    results = search_vector_records(
        query="def main(): pass",
        records=records,
        top_k=2,
        model_name="test-model",
        dimension=32,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "main.py::chunk_0"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0.99
    assert results[0]["content_preview"] == "def main(): pass"
    assert results[0]["source_name"] == "main.py"


def test_vector_search_records_sorts_by_score():
    records = build_vector_records(dimension=32)

    results = search_vector_records(
        query="def main(): pass",
        records=records,
        top_k=2,
        model_name="test-model",
        dimension=32,
    )

    assert results[0]["score"] >= results[1]["score"]
    assert [item["rank"] for item in results] == [1, 2]


def test_vector_search_records_applies_top_k():
    records = build_vector_records(dimension=32)

    results = search_vector_records(
        query="main",
        records=records,
        top_k=1,
        model_name="test-model",
        dimension=32,
    )

    assert len(results) == 1
    assert results[0]["rank"] == 1


def test_vector_search_records_chunk_type_filter():
    records = build_vector_records(dimension=32)

    results = search_vector_records(
        query="main",
        records=records,
        top_k=5,
        model_name="test-model",
        dimension=32,
        chunk_type="code",
    )

    assert len(results) == 1
    assert results[0]["chunk_type"] == "code"


def test_vector_search_records_missing_embedding():
    records = build_vector_records(dimension=32)
    records[0].pop("embedding")

    with pytest.raises(ValueError):
        search_vector_records(
            query="main",
            records=records,
            top_k=5,
            model_name="test-model",
            dimension=32,
        )


def test_vector_search_records_dimension_mismatch():
    records = build_vector_records(dimension=16)

    with pytest.raises(ValueError):
        search_vector_records(
            query="main",
            records=records,
            top_k=5,
            model_name="test-model",
            dimension=32,
        )


def test_vector_search_from_file_success(tmp_path):
    index_path = tmp_path / "vector_index.json"

    save_vector_index(
        records=build_vector_records(dimension=32),
        output_path=str(index_path),
    )

    result = search_vector_index_from_file(
        query="def main(): pass",
        index_path=str(index_path),
        top_k=2,
        model_name="test-model",
        dimension=32,
    )

    assert result["index_path"] == str(index_path)
    assert result["query"] == "def main(): pass"
    assert result["top_k"] == 2
    assert result["model_name"] == "test-model"
    assert result["dimension"] == 32
    assert result["chunk_type"] is None
    assert result["result_count"] == 2
    assert result["results"][0]["chunk_id"] == "main.py::chunk_0"


def test_vector_search_from_file_chunk_type_filter(tmp_path):
    index_path = tmp_path / "vector_index.json"

    save_vector_index(
        records=build_vector_records(dimension=32),
        output_path=str(index_path),
    )

    result = search_vector_index_from_file(
        query="main",
        index_path=str(index_path),
        top_k=5,
        model_name="test-model",
        dimension=32,
        chunk_type="code",
    )

    assert result["chunk_type"] == "code"
    assert result["result_count"] == 1
    assert result["results"][0]["chunk_type"] == "code"


def test_vector_search_invalid_top_k(tmp_path):
    index_path = tmp_path / "vector_index.json"

    save_vector_index(
        records=build_vector_records(dimension=32),
        output_path=str(index_path),
    )

    with pytest.raises(ValueError):
        search_vector_index_from_file(
            query="main",
            index_path=str(index_path),
            top_k=0,
            dimension=32,
        )


def test_vector_search_invalid_dimension(tmp_path):
    index_path = tmp_path / "vector_index.json"

    save_vector_index(
        records=build_vector_records(dimension=32),
        output_path=str(index_path),
    )

    with pytest.raises(ValueError):
        search_vector_index_from_file(
            query="main",
            index_path=str(index_path),
            top_k=3,
            dimension=0,
        )


def test_vector_search_empty_query(tmp_path):
    index_path = tmp_path / "vector_index.json"

    save_vector_index(
        records=build_vector_records(dimension=32),
        output_path=str(index_path),
    )

    with pytest.raises(ValueError):
        search_vector_index_from_file(
            query="   ",
            index_path=str(index_path),
            top_k=3,
            dimension=32,
        )


def test_vector_search_index_not_exists():
    with pytest.raises(FileNotFoundError):
        search_vector_index_from_file(
            query="main",
            index_path="not_exists_vector_index.json",
            top_k=3,
        )
